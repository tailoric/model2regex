from pathlib import Path
from model2regex import *
from itertools import batched
import domain_gen
import re
from argparse import ArgumentParser
from typing import Sequence
choices: Sequence[Callable] = [func for func in domain_gen.IND2FUN.values()]
def gen_dataset(func: Callable[[], str], count: int, *, store_path: Path|None = None) -> list[str]:
    dataset = []
    for _ in range(count):
        dataset.append(func())
    if store_path:
        store_path.write_text('\n'.join(dataset))
    return dataset

def train_multi(data: Path, model_path: Path):
    with data.open() as ds:
        dataset_table = []
        while line := ds.readline():
            splits = [''.join(batch) for batch in batched(line[:-1], n=4)]
            if not dataset_table:
                dataset_table = [[] for _ in range(len(splits))]
            for idx, split in enumerate(splits):
                dataset_table[idx].append(split or '')
        
        model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False)
        datasets = list(map(lambda d: DGADataset(d, real_domains=[]), dataset_table))
        trainer = ModelTrainer(model=model, dataset=datasets[0],model_path=model_path)
        trainer.multi_train(datasets)
        return trainer

def build_regex(dataset: Path, model_path: Path, **kwargs):
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False)
    model.to(kwargs.get('device', 'cuda:0'))
    graphing_path = kwargs.get('graphing_path', Path('graphs'))
    visualize = kwargs.get('visualize', False)
    regex_list = []
    heuristic : Heuristic = kwargs.get('heuristic', Threshold(threshold=0.1, quantile=0.5))
    for num, model_path in enumerate(model_path.iterdir(), start=1):
        model.load_state_dict(torch.load(model_path))
        dfa = DFA(model, heuristic=heuristic)
        dfa.build_tree()
        regex = dfa.build_regex()
        dfa.save_file(Path(graphing_path / f'dfa-{num}.gml.gz'))
        regex_list.append(regex)
        if visualize:
            dfa.visualize_tree(graphing_path/f'dfa-{num}.svg')
    final_regex = ''.join(regex_list)
    return final_regex



if __name__ == "__main__":
    parser = ArgumentParser(description='Run this main file to generate a regular expression from a Language Model.')
    
    parser.add_argument("--steps", help="which additional steps should the program do in this run.", action='append', choices=['gen-dataset', 'train-models', 'test-regex'])
    dataset_group = parser.add_argument_group("dataset", "options for when gen-dataset was added as extra step")
    parser.add_argument('--model-path', type=Path, help="Path where the generated models should be stored if train-models was chosen otherwise the directory that will get read for the models.", required=True)
    dataset_group.add_argument('--domain-generator', help="The generator function to use from domain_gen", choices=[func.__name__ for func in choices])
    dataset_group.add_argument('--store-dataset', action='store_true', help="Set this to specify that the dataset should be stored after generating it.")
    dataset_group.add_argument('--count', type=int, default=100000, help="The amount of entries to create when generating the data.")
    parser.add_argument('data', type=Path, help="Path to the dataset as a txt file with one entry per line, or the path to store the data at when gen-dataset is set.")
    arguments = parser.parse_args()
    dataset_gen_flag = 'gen-dataset' in arguments.steps if arguments.steps else False
    train_model_flag = 'train-models' in arguments.steps if arguments.steps else False
    test_regex_flag = 'test-regex' in arguments.steps if arguments.steps else False
    print(arguments)
    if dataset_gen_flag and not arguments.data:
        raise Exception("Please provide the path to store the data at with --data.")
    if dataset_gen_flag and not arguments.domain_generator:
        raise Exception("Provide the generator you want to use for generating the dataset")

    if dataset_gen_flag:
        func = getattr(domain_gen, arguments.domain_generator)
        gen_dataset(func, arguments.count, store_path=arguments.data)
    if train_model_flag:
        logging.basicConfig()
        train_multi(arguments.data, arguments.model_path)

    regex = build_regex(arguments.data, arguments.model_path, visualize=True)

    if test_regex_flag:
        matched = 0
        count = 0
        print(regex)
        pattern = re.compile(regex)
        with arguments.data.open() as ds:
            while line := ds.readline():
                count += 1
                if pattern.match(line[:-1]):
                    matched += 1

        print(f"Matched {matched:,}/{count:,} ({matched/count:%})")
    print(regex)
    
