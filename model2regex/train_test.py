from model import DGAClassifier, DEFAULT_MODEL_SETTINGS
import pathlib
import unittest
import torch


class TrainedModelTest(unittest.TestCase):

    def setUp(self):
        models_path = pathlib.Path('model2regex/models')
        if not models_path.exists():
            raise unittest.SkipTest("no preset Models found.")
        self.model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
        self.model.load_state_dict(torch.load(models_path / 'model-fold-1.pth'))

    def test_trained_model_input(self):
        _, prediction, _ = self.model('www')
        breakpoint()


if __name__ == "__main__":
    unittest.main()
