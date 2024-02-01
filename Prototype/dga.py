"""
This file works as a repository of dga algorithms to import for the prototype
"""
from random import Random
from typing import Callable

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

def generate_dataset(algorithm: Callable[[str], str], seed: str, size: int) -> list[str]:
    current = algorithm(seed)
    domains = [current]

    for _ in range(size-1):
        new = algorithm(current)
        domains.append(new)
        current = new
    return domains

