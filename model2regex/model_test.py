import unittest
import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from .model import IllegalStartChar, DGAClassifier
from itertools import repeat


class TestData(Dataset):
    def __init__(self, real_domains: list[str], dga_domains: list[str]):
        self.data = list(tuple(zip(real_domains, repeat(0))))
        self.data.extend(list(tuple(zip(dga_domains, repeat(1)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, int]:
        return self.data[idx]


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model: DGAClassifier = DGAClassifier(64, 128, 1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if not torch.cuda.is_available():
            self.model.to('cpu')
            self.model.device = self.device
        else:
            self.model.to('cuda:0')
            self.model.device = self.device

    def test_longer_start_character(self):
        error_msg = "Exception should raise because start_char is longer than 1"
        with self.assertRaises(IllegalStartChar, msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="__")
        exception = exc.exception
        self.assertEqual(str(exception), "The start character should be of length 1.")

    def test_start_character_in_vocabulary(self):
        error_msg = "Exception should raise because character already exists in vocabulary"
        with self.assertRaises(IllegalStartChar, msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="1")
        exception = exc.exception
        self.assertEqual(str(exception), "The start character should not be part of the default vocabulary.")

    def test_charTensor(self):
        google_tensor = torch.tensor(
                [[39, 23, 23, 23, 38,  7, 15, 15,  7, 12,  5, 38,  3, 15, 13,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0]]
                ).permute(1, 0)

        char_tensor = self.model.charTensor(["www.google.com"])
        domain = ''.join(self.model.char2idx[c] for c in char_tensor.permute(1, 0).squeeze().tolist() if c != 0)

        self.assertEqual(tuple(char_tensor.shape), (254, 1))
        self.assertTrue(torch.equal(char_tensor, google_tensor))
        self.assertEqual(domain, "_www.google.com")

    def test_forward_with_string(self):
        _class, _, hidden = self.model(["www.google.com", "reddit.com"], None)
        for item in _class.squeeze().tolist():
            self.assertTrue(0 < item <= 1)
        self.assertTupleEqual(hidden.size(), (1, 2, 128))

    def test_forward_with_tensors(self):
        google_tensor = torch.tensor(
                [39, 23, 23, 23, 38,  7, 15, 15,  7, 12,  5, 38,  3, 15, 13,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0]
                )
        reddit_tensor = torch.tensor(
                [39, 18, 5, 4, 4, 9, 20, 38, 3, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0]
                )
        _class, prediction, hidden = self.model(torch.stack((google_tensor, reddit_tensor)).permute(1, 0).to(self.device), None)
        for item in _class.squeeze().tolist():
            self.assertTrue(0 < item <= 1)
        self.assertTupleEqual(hidden.size(), (1, 2, 128))

    def test_with_data_loader(self):
        dataset = TestData(['www.google.com', 'reddit.com', 'amazon.de'], ['abcdefg.net', 'defghijkl.net', 'mnopqrstu.net'])
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch, (x, y) in enumerate(loader):
            cls, prediction, hidden = self.model(x, None)
            self.assertTupleEqual(hidden.size(), (1, 2, 128))
            for item in cls.squeeze().tolist():
                self.assertTrue(0 < item <= 1)

    def test_types_of_predict(self):
        prediction = self.model.predict("")
        self.assertIsInstance(prediction, str)
        self.assertRegex(prediction, r"^[a-z0-9._-]+<END>")

    def test_types_of_predict_next_token(self):
        token, distribution = self.model.predict_next_token("")
        self.assertIsInstance(token, int)
        self.assertIsInstance(distribution, Categorical)


if __name__ == "__main__":
    unittest.main()
