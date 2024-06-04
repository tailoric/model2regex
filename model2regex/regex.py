import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import TypedDict
from model2regex.model import DEFAULT_MODEL_SETTINGS, DGAClassifier
import torch
UP = "\x1B[3A"
CLR = "\x1B[0K"

class Node(TypedDict):
    item: str
    depth: int


class DFA:
    def __init__(self, model: DGAClassifier, root_starter: str = "", threshold: float = 0.4):
        root_node: Node = {'item': root_starter, 'depth': 0}
        self.graph = nx.DiGraph()
        self.graph.add_node(0, **root_node)
        self.model = model
        self.model.eval()
        self.threshold = threshold
        self.store_path = Path("graphs")

    def build_tree(self, store = False) -> None:
        """
        builds the tree for the DFA, by adding to the DiGraph
        """
        char_map = self.model.char2idx
        nodes_to_visit: list[tuple[int,Node]] = [(0,self.graph.nodes[0])]
        id_counter = 0
        end_nodes = 0
        while nodes_to_visit:
            node_id, data = nodes_to_visit.pop(0)
            depth = data.get('depth')
            starter: list[str|None] = [data.get('item')]
            parent  = list(self.graph.predecessors(node_id))
            while parent:
                parent_node = self.graph.nodes[parent[0]]
                starter.append(parent_node.get('item'))
                parent: list[int] = list(self.graph.predecessors(parent[0]))
            starter = "".join(reversed(starter))
            _, distribution = self.model.predict_next_token(starter)
            mask = distribution.probs > self.threshold
            if torch.any(mask):
                indices = torch.argwhere(mask).squeeze().tolist()
            else:
                indices = torch.topk(distribution.probs, 3).indices.squeeze().tolist()
            if not isinstance(indices, list):
                indices = [indices]
            for idx in indices:
                new_node : Node = {'item': char_map[idx], 'depth': depth + 1}
                new_node_id = id_counter + 1
                self.graph.add_node(new_node_id, **new_node)
                self.graph.add_edge(node_id, new_node_id)
                if idx != 0:
                    nodes_to_visit.append((new_node_id, new_node))
                else:
                    end_nodes += 1
                id_counter += 1

            print(f"{UP}nodes to visit: {len(nodes_to_visit):,}, current starter: {starter}{CLR}\n"+
                  f"tree nodes: {len(self.graph):,}, end nodes: {end_nodes} depth: {depth}{CLR}\n")

        if store:
            self.store_path.mkdir(exist_ok=True)
            nx.write_gml(self.graph, self.store_path / 'graph.gml.gz')

    def visualize_tree(self) -> None:
        layout = nx.bfs_layout(self.graph, 0)
        nx.draw(self.graph, labels=dict(self.graph.nodes(data='item')), pos=layout, with_labels=True)
        plt.show()

if __name__ == "__main__":
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
    model.load_state_dict(torch.load('models/model-fold-1.pth'))
    model.to("cuda:0")
    dfa = DFA(model, root_starter="www.go", threshold=0.4)
    dfa.build_tree(store=True)
    dfa.visualize_tree()
