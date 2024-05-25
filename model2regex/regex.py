from typing import Self
from model2regex.model import DGAClassifier
import torch
UP = "\x1B[3A"
CLR = "\x1B[0K"

class Node:
    """
    Helper class for a tree node of the DFA
    """

    def __init__(self, parent: Self | None, item: str, depth: int = 0):
        self.parent = parent
        self._children: list[Self] = []
        self.item: str = item
        self.depth = depth

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
        depth = 0
        nodes_to_visit = [self.root]
        char2idx = self.model.char2idx
        tree_nodes = 0
        end_nodes = 0
        print("\n\n")
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            depth = current_node.depth
            parent = current_node.parent
            starter = current_node.item
            while parent:
                starter += parent.item
                parent = parent.parent
            starter = "".join(reversed(starter))
            print(f"{UP}nodes to visit: {len(nodes_to_visit):,}, current starter: {starter}{CLR}\n"+
                  f"tree nodes: {tree_nodes:,}, end nodes: {end_nodes}, depth: {depth}{CLR}\n")
            _, distribution = self.model.predict_next_token(starter)
            mask = distribution.probs > self.threshold
            if any(mask):
                indices = torch.argwhere(mask).squeeze().tolist()
                if not isinstance(indices, list):
                    indices = [indices]
                for idx in indices:
                    new_node = Node(parent=current_node, item=char2idx[idx], depth=depth+1)
                    current_node.add_child(new_node)
                    tree_nodes += 1
                    if idx != 0:
                        nodes_to_visit.append(new_node)
                    else:
                        end_nodes += 1
            else:
                for idx in torch.topk(distribution.probs, 3).indices.squeeze().tolist():
                    char = char2idx[idx]
                    if char == self.model.start_char:
                        continue
                    new_node = Node(parent=current_node, item=char, depth=depth+1)
                    current_node.add_child(new_node)
                    tree_nodes += 1
                    if idx != 0:
                        nodes_to_visit.append(new_node)
                    else:
                        end_nodes += 1
