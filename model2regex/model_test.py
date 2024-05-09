from model import IllegalStartChar, DGAClassifier
import unittest
import torch


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model: DGAClassifier = DGAClassifier(64, 128, 1)

    def test_longer_start_character(self):
        error_msg = "Exception should raise because start_char is longer than 1"
        with self.assertRaises(IllegalStartChar, msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="__")
        exception = exc.exception
        self.assertEqual(str(exception), "The start character should be of length 1")

    def test_start_character_in_vocabulary(self):
        error_msg = "Exception should raise because character already exists in vocabulary"
        with self.assertRaises(IllegalStartChar, msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="1")
        exception = exc.exception
        self.assertEqual(str(exception), "The start character should not be part of the default vocabulary")

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
        domain = ''.join(self.model.char2idx[c] for c in char_tensor.permute(1,0).squeeze().tolist() if c != 0)

        self.assertEqual(tuple(char_tensor.shape), (254, 1))
        self.assertTrue(torch.equal(char_tensor, google_tensor))
        self.assertEqual(domain, "_www.google.com")

    def test_forward_with_string(self):
        _class, hidden = self.model(["www.google.com", "reddit.com"], None)
        for item in _class.squeeze().tolist():
            self.assertTrue(0 < item <= 1)
        self.assertTupleEqual(tuple(hidden.shape), (1, 2, 128))


if __name__ == "__main__":
    unittest.main()
