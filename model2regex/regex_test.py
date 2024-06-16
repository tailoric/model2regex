from .model import DGAClassifier, DEFAULT_MODEL_SETTINGS
from .regex import DFA, Node
import pathlib
import unittest
import torch
import logging
import tempfile
import networkx as nx


class RegexGenTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_directory = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.test_directory.cleanup()

    def setUp(self):
        models_path = pathlib.Path('models')
        if not models_path.exists():
            raise unittest.SkipTest("no preset Models found.")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
        self.model.load_state_dict(torch.load(models_path / 'model-fold-2.pth', map_location=self.device))
        self.model.device = self.device
        self.model.to(self.device)
        path = pathlib.Path(self.test_directory.name)
        self.test_dfa = DFA(self.model, threshold=0.4, store_path=path, root_starter="www.google.")

    def test_build_tree(self):
        self.test_dfa.build_tree(store=True)
        self.assertTrue((self.test_dfa.store_path / 'graph.gml.gz').exists())

    def test_loading_dfa_tree(self):
        self.test_dfa.build_tree()
        new_dfa = DFA(self.model, threshold=0.4, store_path=pathlib.Path(self.test_directory.name), root_starter="")
        new_dfa.load_file(new_dfa.store_path / 'graph.gml.gz')
        start_node = new_dfa.graph.nodes[0]
        self.assertEqual(start_node.get('item'), 'www.google.')
        self.assertEqual(start_node.get('depth'), 0)
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
