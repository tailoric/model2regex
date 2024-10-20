#!/usr/bin/python3
import random
import argparse

from . import exrex

def generate_url_scheme_1():
    return exrex.getone(r'[a-c]{2}[ab]{2}-domain\.com')

def generate_url_scheme_2():
    return exrex.getone(r'[a-d]{4}-domain\.com')

# simple way of creating "stateless" domains
def simpleExreg():
    return exrex.getone(r'[0-9]{3}[a-c]{2,5}[x-z]{,4}\.(evil|knievel)\.com')

# one way of having a state in the generation process 
def statefulExreg():
    if random.random() <= 0.5:
        return exrex.getone(r'[0-9]{5}\.evil\.[a-c]{1,2}\.com')
    else:
        return exrex.getone(r'[0-9]{5}\.knievel\.[a-c]{3,4}\.com')

def statefulExregSameLength():
    if random.random() <= 0.5:
        return exrex.getone(r'[0-9]{5}\.evil\.[a-c]{4}\.com')
    else:
        return exrex.getone(r'[0-9]{3}\.knievel\.[a-c]{3}\.com')

IND2FUN = {1: generate_url_scheme_1, 2: generate_url_scheme_2,
           3: simpleExreg, 4: statefulExreg, 5: statefulExregSameLength}

def main():
    schemes = sorted(IND2FUN.keys())
    parser = argparse.ArgumentParser(description='Generate URLs using different schemes.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", '--scheme', type=int, choices=schemes, help='Choose the URL generation scheme', default=max(schemes))
    parser.add_argument("-c", '--count', type=int, help='Choose number of URL generated', default=10)

    args = parser.parse_args()

    if args.scheme not in IND2FUN:
        print(f"Invalid scheme selected. Please choose {schemes}.")
    else:
        print("\n".join([IND2FUN[args.scheme]() for _ in range(args.count)]))

if __name__ == "__main__":
    main()

