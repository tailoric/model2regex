from .model import DGAClassifier, DEFAULT_MODEL_SETTINGS
from .regex import DFA
import pathlib
import unittest
import torch
import logging


class RegexGenTest(unittest.TestCase):

    def setUp(self):
        models_path = pathlib.Path('models')
        if not models_path.exists():
            raise unittest.SkipTest("no preset Models found.")
        self.model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
        self.model.load_state_dict(torch.load(models_path / 'model-fold-2.pth'))
        self.model.to("cuda:0")

    @unittest.skip("Not ready yet")
    def test_build_tree(self):
        dfa = DFA(self.model, threshold=0.4)
        dfa.build_tree()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
