import dga
import unittest


class TestDGA(unittest.TestCase):

    def test_generate_dataset(self):
        dataset = dga.generate_dataset(dga.banjori, 'earnestnessbiophysicalohax')
        self.assertEqual(len(dataset), 2_000_000)


if __name__ == "__main__":
    unittest.main()
