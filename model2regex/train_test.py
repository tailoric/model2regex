from .model import DGAClassifier, DEFAULT_MODEL_SETTINGS
import pathlib
import unittest
import torch


class TrainedModelTest(unittest.TestCase):

    def setUp(self):
        models_path = pathlib.Path('models')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if not models_path.exists():
            raise unittest.SkipTest("no preset Models found.")
        self.model = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
        self.model.load_state_dict(torch.load(models_path / 'model-fold-2.pth'))
        self.model.to(self.device)
        self.model.device = self.device

    def test_trained_model_input(self):
        _ = self.model.predict("www")


if __name__ == "__main__":
    unittest.main()
