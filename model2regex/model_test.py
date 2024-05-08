from model import IllegalStartChar, DGAClassifier
import unittest


class TestModel(unittest.TestCase):

    def test_longer_start_character(self):
        error_msg = "Exception should raise because start_char is longer than 1"
        with self.assertRaises(IllegalStartChar, msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="__")
        exception = exc.exception
        self.assertEqual(str(exception), "The start character should be of length 1")

    def test_start_character_in_vocabulary(self):
        error_msg = "Exception should raise because character already exists in vocabulary"
        with self.assertRaises(IllegalStartChar,msg=error_msg) as exc:
            DGAClassifier(64, 128, 1, start_char="1")
        exception = exc.exception
        self.assertEqual(str(exception),"The start character should not be part of the default vocabular")


if __name__ == "__main__":
    unittest.main()
