import argparse
IND2FUN = {1: generate_url_scheme_1, 2: generate_url_scheme_2,
           3: simpleExreg, 4: statefulExreg}

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

