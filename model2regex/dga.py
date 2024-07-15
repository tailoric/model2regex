"""
This file works as a repository of dga algorithms to import for the prototype
"""
from random import Random
from typing import Callable, Literal, Sequence
from torch.utils.data import Dataset
from itertools import repeat, batched
from pathlib import Path
import pandas as pd

real_domains = pd.read_csv(Path('data/top-1m.csv'), header=None).values[:, 1]


class DGADataset(Dataset):
    def __init__(self, dgas: list[str], real_domains: list[str]):
        self.data = list(zip(dgas, repeat(1.0)))
        self.data.extend(list(zip(real_domains, repeat(0.0))))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, Literal[0, 1]]:
        return self.data[idx]

def simple_dga(seed: str) -> str:
    rand = Random(seed)
    digits = '0123456789'
    return 'a' + ''.join(rand.choice(digits) for _ in range(rand.randint(10, 20))) + rand.choice(['.net', '.com','.xyz'])


def banjori(domain: str) -> str:
    def map_to_lowercase_letter(s):
        return ord('a') + ((s - ord('a')) % 26)
    dl = [ord(x) for x in list(domain)]
    dl[0] = map_to_lowercase_letter(dl[0] + dl[3])
    dl[1] = map_to_lowercase_letter(dl[0] + 2*dl[1])
    dl[2] = map_to_lowercase_letter(dl[0] + dl[2] - 1)
    dl[3] = map_to_lowercase_letter(dl[1] + dl[2] + dl[3])
    return ''.join([chr(x) for x in dl])


def generate_split_data(
        algorithm: Callable[[str], str],
        seed: str,
        size = 2_000_00,
        split_size = 4
        ) -> Sequence[Dataset]:
    datasets = []
    splits = [[''.join(batch)] for batch in batched(seed, n=split_size)]
    datasets.extend(splits)
    while len(datasets[0]) < size:
        seed = algorithm(seed)
        splits = [''.join(batch) for batch in batched(seed, n=split_size)]
        for idx,split in enumerate(splits):
            datasets[idx].append(split)
    return list(map(lambda d: DGADataset(d, real_domains=[]), datasets))

def generate_dataset(algorithm: Callable[[str], str],
                     seed: str,
                     size: int = 2_000_000,
                     real_domains: list[str] = real_domains
                     ) -> Dataset:
    """
    Generate a DGA dataset that also contains real domains for the other class

    Parameter
    ---------
    algorithm: Callable[[str], str]
        Function that will generate the DGA domains should be a function that accepts a single seed 
        and returns a domain.
    seed: str
        The initial seed of the DGA algorithm.
    size: int
        size of the final dataset (dga + real domains)
        defaults to 2 million.
    real_domains: list[str]
        a list of real domains as the benign class, defaults to top 1 million most visited domains dataset.
        if this list is empty then generate a dataset only containing DGAs
    """
    if size > 2 * len(real_domains) and len(real_domains) > 0:
        size = 2 * len(real_domains)
    current = algorithm(seed)
    domains = [current]
    real_domains = real_domains[:size//2]
    while len(domains) < (size - len(real_domains)):
        domains.append(algorithm(current))
        current = algorithm(current)
    return DGADataset(domains, real_domains)
