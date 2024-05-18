import dga
import unittest


class TestDGA(unittest.TestCase):

    def test_generate_dataset(self):
        dataset = dga.generate_dataset(dga.banjori, 'earnestnessbiophysicalohax')
        self.assertEqual(len(dataset), 2_000_000)

    def test_smaller_dataset(self):
        dataset = dga.generate_dataset(dga.banjori, 'earnestnessbiophysicalohax', size=50_000)
        self.assertEqual(len(dataset), 50_000)
        self.assertEqual(int(sum(d[1] for d in dataset.data)), 25_000, "Half of the data should be in class 1")

    def test_dataset_bigger_than_2mil(self):
        dataset = dga.generate_dataset(dga.banjori, 'earnestnessbiophysicalohax', size=3_000_000)
        self.assertEqual(len(dataset), 2_000_000)


if __name__ == "__main__":
    unittest.main()
