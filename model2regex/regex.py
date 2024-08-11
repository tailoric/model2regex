from math import log
from random import choice
from itertools import repeat
import networkx as nx
from pathlib import Path
from typing import NotRequired, Tuple, TypedDict, Sequence, Protocol

from torch.distributions import Categorical
from model2regex.model import DEFAULT_MODEL_SETTINGS, DGAClassifier
import re
import torch

UP = "\x1B[3A"
CLR = "\x1B[0K"

class Node(TypedDict):
    """
    dict struct for typing node information in the DFA
    """
    item: str
    type: str
    depth: int
    classification: NotRequired[float]

class Heuristic(Protocol):
    """
    base class that defines the heuristic strategy for building the regex tree.
    """
    
    def next_node(self, distribution: Categorical) -> Tuple[Sequence[int], str]:
        """
        return the indices for the char_map of the distribution to add to the tree.
        """
        ...

class Threshold(Heuristic):

    def __init__(self, threshold: float = 0.4, quantile: float = 0.5) -> None:
        self.threshold = threshold
        self.quantile = quantile

    def next_node(self, distribution: Categorical) -> Tuple[Sequence[int], str]:
        mask = distribution.probs > self.threshold
        if torch.any(mask):
            indices = torch.argwhere(mask).squeeze().tolist()
            item_type = 'simple'
        else:
            mask = distribution.probs >= torch.quantile(distribution.probs, self.quantile, interpolation='nearest')
            indices = torch.argwhere(mask).squeeze().tolist()
            item_type = 'simple'
        return indices, item_type

class Entropy(Heuristic):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def next_node(self, distribution: Categorical) -> Tuple[Sequence[int], str]:
        entropy = - torch.sum(distribution.probs * distribution.probs.log()) # type: ignore
        if entropy < self.threshold:
            item_type = 'simple'
            mask = distribution.probs > torch.quantile(distribution.probs, 0.75, interpolation='linear')
        else:
            item_type = 'simple'
            mask = distribution.probs >= torch.quantile(distribution.probs, 0.30, interpolation='nearest')
        return torch.argwhere(mask).squeeze().tolist(), item_type


class DFA:
    def __init__(self, model: DGAClassifier, store_path = Path("graphs"), root_starter: str = "", heuristic: Heuristic = Threshold()):
        root_node: Node = {'item': root_starter, 'depth': 0, 'type': 'root'}
        self.graph = nx.DiGraph()
        self.graph.add_node(0, **root_node)
        self.model = model
        self.model.eval()
        self.heuristic = heuristic
        self.store_path = store_path

    def build_tree(self, store = False) -> None:
        """
        builds the tree for the DFA, by adding to the DiGraph in self.graph
        """
        char_map = self.model.char2idx
        nodes_to_visit: list[tuple[int,Node]] = [(0,self.graph.nodes[0])]
        id_counter = 0
        end_nodes = 0
        while nodes_to_visit:
            node_id, data = nodes_to_visit.pop(0)
            depth = data.get('depth')
            root_path_symbols = [choice(data.get('item')) if data.get('type') == 'group' else data.get('item')]
            parent  = list(self.graph.predecessors(node_id))
            while parent:
                parent_node: Node = self.graph.nodes[parent[0]]
                if parent_node['type'] in ('simple','root'):
                    root_path_symbols.append(parent_node.get('item'))
                else:
                    root_path_symbols.append(choice(parent_node.get('item')))
                parent: list[int] = list(self.graph.predecessors(parent[0]))
            starter = "".join(reversed(root_path_symbols))
            _, distribution = self.model.predict_next_token(starter)
            indices, item_type = self.heuristic.next_node(distribution)
            if not isinstance(indices, list):
                indices = [indices]
            if item_type == "group":
                item = [char_map[x] for x in indices]
                new_node : Node = { 'item': item, 'depth': depth + 1, 'type':item_type}
                new_node_id = id_counter + 1
                self.graph.add_node(new_node_id, **new_node)
                mean = torch.mean(distribution.probs[indices])
                self.graph.add_edge(node_id, new_node_id, probability=round(mean.item(), ndigits=2))
                nodes_to_visit.append((new_node_id, new_node))
                id_counter += 1
            else:
                for idx in indices:
                    new_node : Node = {'item': char_map[idx], 'depth': depth + 1, 'type': 'simple'}
                    new_node_id = id_counter + 1
                    if idx != 0:
                        nodes_to_visit.append((new_node_id, new_node))
                    else:
                        x, _, _ = self.model([starter], None)
                        new_node['classification'] = x.round().item()
                        end_nodes += 1
                    self.graph.add_node(new_node_id, **new_node)
                    self.graph.add_edge(node_id, new_node_id, probability=round(distribution.probs[idx].item(), ndigits=2))
                    id_counter += 1

            #print(f"{UP}nodes to visit: {len(nodes_to_visit):,}, current starter: {starter}{CLR}\n"+
            #      f"tree nodes: {len(self.graph):,}, end nodes: {end_nodes} depth: {depth}, entropy {-torch.sum(distribution.probs * distribution.probs.log())}{CLR}\n")

        if store:
            self.save_file()
    def simplify_tree(self, iterations:int = 3):
        layers = nx.bfs_layers(self.graph, 0)
        layers = reversed(list(layers))
        for layer in layers:
            for node in layer:
                if node not in self.graph:
                    continue
                parent = next(self.graph.predecessors(node))
                successors = list(self.graph.successors(parent))
                if len(successors) > 1:
                    KL = 0
                    for child in successors:
                        edge = self.graph.edges[parent,child]
                        KL += edge['probability'] * log((1/len(successors))/edge['probability'])
                    if KL < 0.1:
                        old_node = self.graph.nodes[node]
                        new_node: Node = {'item': [self.graph.nodes[child]['item'] for child in successors], 
                                          'type': 'group',
                                          'depth': old_node['depth'] }
                        outgoing = []
                        for child in successors:
                            outgoing.extend(self.graph.neighbors(child))
                        self.graph.remove_nodes_from(successors)
                        self.graph.add_node(node, **new_node)
                        self.graph.add_edges_from(zip(repeat(node), outgoing))
                        self.graph.add_edge(parent, node)


    def load_file(self, file_path: Path | None):
        """
        load file from the file_path provided
        """
        if not file_path:
            file_path = self.store_path / 'graph.gml.gz'
        self.graph = nx.read_gml(file_path, label='id')

    def save_file(self, store_path: str | Path | None = None, /):
        """
        save the current dfa as a file called graph.gml.gz at the store path
        """
        if not store_path:
            store_path = self.store_path / 'graph.gml.gz'
        else:
            store_path = Path(store_path)
        if store_path.is_dir():
            store_path.mkdir(exist_ok=True)
        else:
            store_path.parent.mkdir(exist_ok=True)
        nx.write_gml(self.graph, store_path)

    def visualize_tree(self, store_path: Path | str = Path('graphs/tree.svg')) -> None:
        """
        generate an svg visualization using pydot
        """
        gp = nx.nx_pydot.to_pydot(self.graph)
        for node in gp.get_nodes():
            node.set_label(node.get('item'))
        for edge in gp.get_edges():
            edge.set_label(edge.get('probability'))

        gp.write_svg(store_path)

    def build_regex(self) -> str: 
        """
        build a regex from the current DFA tree.
        """
        return self._build_from_subgraph(self.graph, 0)

    def _build_from_subgraph(self, subgraph, source) -> str:
        regex_str = ""
        nb = nx.neighbors(subgraph, source)
        for node in nb:
            data = self.graph.nodes[node]
            degree = (self.graph.degree[node] - 1) # remove one because of incoming parent node
            if data['item'] == "<END>":
                continue
            if degree > 1:
                regex_str += f"{data['item']}({self._build_from_subgraph(self.graph, node)})|"
            elif degree == 1:
                regex_str += f"{data['item']}{self._build_from_subgraph(self.graph, node)}|"
            else:
                regex_str += f"{data['item']}|"
        if regex_str and regex_str[-1] == "|":
            regex_str = regex_str[:-1]
        return regex_str

if __name__ == "__main__":
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
    model.load_state_dict(torch.load('models_backwards/model-backwards.pth'))
    model.to("cuda:0")
    dfa = DFA(model, root_starter="", heuristic=Threshold())
    dfa.build_tree(store=True)
    dfa.load_file(file_path=Path('graphs/graph.gml.gz'))
    regex = dfa.build_regex()
    #regex = ''.join(reversed(regex))
    print(regex)
    dga_regex = re.compile(regex)
    matched = 0
    with open('banjori.txt', 'r') as test_file:
        lines = test_file.readlines()
        for line in lines:
            rev = ''.join(reversed(line[:-1]))
            match = dga_regex.match(rev)
            if match:
                matched += 1
    print(f"matched {matched} out of {len(lines)} ({matched/len(lines):%})")
    dfa.visualize_tree()
