import networkx as nx
from pathlib import Path
from typing import NotRequired, TypedDict, Sequence, Protocol

from torch.distributions import Categorical
from model2regex.model import DEFAULT_MODEL_SETTINGS, DGAClassifier
import torch

UP = "\x1B[3A"
CLR = "\x1B[0K"

class Node(TypedDict):
    """
    dict struct for typing node information in the DFA
    """
    item: str
    depth: int
    classification: NotRequired[float]

class Heuristic(Protocol):
    """
    base class that defines the heuristic strategy for building the regex tree.
    """
    
    def next_node(self, distribution: Categorical) -> Sequence[int]:
        """
        return the indices for the char_map of the distribution to add to the tree.
        """
        ...

class Threshold(Heuristic):

    def __init__(self, threshold: float = 0.4, topk: int = 3) -> None:
        self.threshold = threshold
        self.topk = topk

    def next_node(self, distribution: Categorical) -> Sequence[int]:
        mask = distribution.probs > self.threshold
        if torch.any(mask):
            indices = torch.argwhere(mask).squeeze().tolist()
        else:
            indices = torch.topk(distribution.probs, self.topk).indices.squeeze().tolist()
        return indices

class Entropy(Heuristic):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def next_node(self, distribution: Categorical) -> Sequence[int]:
        entropy = - torch.sum(distribution.probs * distribution.probs.log()) # type: ignore
        if entropy < self.threshold:
            mask = distribution.probs >= torch.quantile(distribution.probs, 0.75, interpolation='nearest')
        else:
            mask = distribution.probs >= torch.quantile(distribution.probs, 0.25, interpolation='nearest')
        return torch.argwhere(mask).squeeze().tolist()


class DFA:
    def __init__(self, model: DGAClassifier, store_path = Path("graphs"), root_starter: str = "", heuristic: Heuristic = Threshold()):
        root_node: Node = {'item': root_starter, 'depth': 0}
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
            root_path_symbols: list[str] = [data.get('item')]
            parent  = list(self.graph.predecessors(node_id))
            while parent:
                parent_node: Node = self.graph.nodes[parent[0]]
                root_path_symbols.append(parent_node.get('item'))
                parent: list[int] = list(self.graph.predecessors(parent[0]))
            starter = "".join(reversed(root_path_symbols))
            _, distribution = self.model.predict_next_token(starter)
            indices = self.heuristic.next_node(distribution)
            if not isinstance(indices, list):
                indices = [indices]
            for idx in indices:
                new_node : Node = {'item': char_map[idx], 'depth': depth + 1}
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

            print(f"{UP}nodes to visit: {len(nodes_to_visit):,}, current starter: {starter}{CLR}\n"+
                  f"tree nodes: {len(self.graph):,}, end nodes: {end_nodes} depth: {depth}, entropy {-torch.sum(distribution.probs * distribution.probs.log())}{CLR}\n")

        if store:
            self.save_file()

    def load_file(self, file_path: Path | None):
        """
        load file from the file_path provided
        """
        if not file_path:
            file_path = self.store_path / 'graph.gml.gz'
        self.graph = nx.read_gml(file_path, label='id')

    def save_file(self):
        """
        save the current dfa as a file called graph.gml.gz at the store path
        """
        self.store_path.mkdir(exist_ok=True)
        nx.write_gml(self.graph, self.store_path / 'graph.gml.gz')

    def visualize_tree(self) -> None:
        """
        generate an svg visualization using pydot
        """
        gp = nx.nx_pydot.to_pydot(self.graph)
        for node in gp.get_nodes():
            node.set_label(node.get('item'))
        for edge in gp.get_edges():
            edge.set_label(edge.get('probability'))

        gp.write_svg(Path('graphs/tree.svg'))

    def build_regex(self) -> str: 
        """
        build a regex from the current DFA tree.
        """
        order = nx.dfs_tree(self.graph, source=0)
        # a stack of symbols we keep track of so we can apply them whenever we get back out from an end node
        end_symbol_stack = []
        regex_str = ""
        previous_depth = 0
        for node in order:
            data = self.graph.nodes[node]
            item = data['item']
            num_child = len(self.graph[node])
            # we came back out of an end node and need to apply all the closing brackets
            if previous_depth > data['depth'] and regex_str[-1] == ")":
                regex_str += end_symbol_stack.pop()
                # we apply all possible closing brackets and a | to the regex string
                while end_symbol_stack and regex_str[-1] != "|":
                    regex_str += end_symbol_stack.pop()
            if item == "<END>" and end_symbol_stack:
                regex_str += end_symbol_stack.pop()
            else:
                regex_str += item
            if num_child > 1:
                regex_str += "("
                end_symbol_stack.extend(")"+ ("|" * (num_child - 1)))
            previous_depth = data['depth']
        regex_str += "".join(end_symbol_stack)
        return regex_str
                


if __name__ == "__main__":
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
    model.load_state_dict(torch.load('models_lm/model-fold-1.pth'))
    model.to("cuda:0")
    dfa = DFA(model, root_starter="", heuristic=Entropy(threshold=2.8))
    dfa.build_tree(store=True)
    dfa.load_file(file_path=Path('graphs/graph.gml.gz'))
    print(dfa.build_regex())
    dfa.visualize_tree()
