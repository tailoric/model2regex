import matplotlib.pyplot as plt
import networkx as nx
from typing import Self
from model2regex.model import DEFAULT_MODEL_SETTINGS, DGAClassifier
import torch
UP = "\x1B[3A"
CLR = "\x1B[0K"

class Node:
    """
    Helper class for a tree node of the DFA
    """

    def __init__(self, parent: Self | None, item: str, id: int = 0):
        self.parent = parent
        self._children: list[Self] = []
        self.item: str = item
        self.id = id

    def add_child(self, node: Self) -> None:
        self._children.append(node)

    @property
    def children(self):
        return self._children

    def __repr__(self) -> str:
        return f"<Node, item = {self.item}, children={len(self._children)}>"

class DFA:
    def __init__(self, model: DGAClassifier, threshold: float = 0.4):
        self.root = Node(parent=None, item="")
        self.model = model
        self.model.eval()
        self.threshold = threshold

    def build_tree(self) -> None:
        nodes_to_visit = [self.root]
        char2idx = self.model.char2idx
        tree_nodes = 0
        end_nodes = 0
        print("\n\n")
        node_id = 0
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            node_id += 1
            parent = current_node.parent
            starter = current_node.item
            while parent:
                starter += parent.item
                parent = parent.parent
            starter = "".join(reversed(starter))
            print(f"{UP}nodes to visit: {len(nodes_to_visit):,}, current starter: {starter}{CLR}\n"+
                  f"tree nodes: {tree_nodes:,}, end nodes: {end_nodes}{CLR}\n")
            _, distribution = self.model.predict_next_token(starter)
            mask = distribution.probs > self.threshold
            if any(mask):
                indices = torch.argwhere(mask).squeeze().tolist()
                if not isinstance(indices, list):
                    indices = [indices]
                for idx in indices:
                    new_node = Node(parent=current_node, item=char2idx[idx], id=node_id)
                    current_node.add_child(new_node)
                    tree_nodes += 1
                    if idx != 0:
                        nodes_to_visit.append(new_node)
                    else:
                        end_nodes += 1
            else:
                indices = torch.topk(distribution.probs, 2).indices.squeeze().tolist()
                if not isinstance(indices, list):
                    indices = [indices]
                for idx in indices:
                    char = char2idx[idx]
                    if char == self.model.start_char:
                        continue
                    new_node = Node(parent=current_node, item=char, id=node_id)
                    current_node.add_child(new_node)
                    tree_nodes += 1
                    if idx != 0:
                        nodes_to_visit.append(new_node)
                    else:
                        end_nodes += 1

    def visualize_tree(self) -> None:
        vertices = []
        edges = []
        nodes_to_visit = [self.root]
        labels = {}
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            vertices.append([current_node.id, {"label": current_node.item}])
            labels[current_node.id] = current_node.item
            nodes_to_visit.extend(current_node.children)
            if current_node.parent:
                edges.append([current_node.parent.id, current_node.id])

        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        layout = nx.bfs_layout(G, 0)
        nx.draw(G, pos=layout, labels=labels, with_labels=True)
        plt.show()

if __name__ == "__main__":
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
    model.load_state_dict(torch.load('models/model-fold-1.pth'))
    model.to("cuda:0")
    model.eval()
    dfa = DFA(model, threshold=0.4)
    #a = Node(item='a', parent=dfa.root, id=1)
    #b = Node(item='b', parent=dfa.root, id=2)
    #c = Node(item='c', parent=a, id=3)
    #b2 = Node(item='b', parent=b, id=4)
    #c2 = Node(item='c', parent=b, id=5) 
    #c3 = Node(item='c', parent=c, id=6) 
    #dfa.root.add_child(a)
    #dfa.root.add_child(b)
    #a.add_child(c)
    #c.add_child(c3)
    #b.add_child(b2)   
    #b2.add_child(c2)
    dfa.build_tree()
    dfa.visualize_tree()
