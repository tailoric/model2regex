from .model import DGAClassifier, DEFAULT_MODEL_SETTINGS
from .regex import DFA, Node, Threshold
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
        self.test_dfa = DFA(self.model, heuristic=Threshold(), store_path=path, root_starter="www.google.")

    def test_build_tree(self):
        self.test_dfa.build_tree(store=True)
        self.assertTrue((self.test_dfa.store_path / 'graph.gml.gz').exists())

    def test_loading_dfa_tree(self):
        self.test_dfa.build_tree()
        new_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        new_dfa.load_file(new_dfa.store_path / 'graph.gml.gz')
        start_node = new_dfa.graph.nodes[0]
        self.assertEqual(start_node.get('item'), 'www.google.')
        self.assertEqual(start_node.get('depth'), 0)
        

    def test_build_regex_max_two_children_max_depth_two(self):
        simple_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        simple_dfa.load_file(pathlib.Path("test/test_simple_nodes_two_children.gml"))
        regex = simple_dfa.build_regex()
        self.assertEqual("a(a|b)|b(a|b)|cc", regex)

    def test_build_regex_max_three_children_max_depth_two(self):
        simple_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        simple_dfa.load_file(pathlib.Path("test/test_simple_nodes_three_children_depth_two.gml"))
        regex = simple_dfa.build_regex()
        self.assertEqual("a(a|b|c)|b(a|b)|cc", regex)

    def test_build_regex_max_two_children_depth_three(self):
        simple_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        simple_dfa.load_file(pathlib.Path("test/test_simple_nodes_depth_three.gml"))
        regex = simple_dfa.build_regex()
        self.assertEqual("a(a(a|b)|b(a|b))|b(a|b)", regex)

    def test_build_regex_max_two_children_depth_four_center(self):
        simple_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        simple_dfa.load_file(pathlib.Path("test/test_simple_nodes_depth_four_middle.gml"))
        regex = simple_dfa.build_regex()
        self.assertEqual("a(a(a|b)|b(a(a|b)|b))|b(a|b)", regex)

    def test_simplify_regex(self):
        simple_dfa = DFA(self.model, heuristic=Threshold(), store_path=pathlib.Path(self.test_directory.name), root_starter="")
        simple_dfa.load_file(pathlib.Path("test/test_simple_nodes_two_children.gml"))
        simple_dfa.simplify_tree(iterations=2)
        regex = simple_dfa.build_regex()
        self.assertEqual("[ab][ab]|cc", regex)

        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
