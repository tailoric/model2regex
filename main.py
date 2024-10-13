from collections.abc import Iterable
from pathlib import Path
from model2regex import *
from itertools import batched
from collections import defaultdict
import domain_gen
import re
from argparse import ArgumentParser
from typing import Sequence
import sqlite3
import pandas as pd

choices: Sequence[Callable] = [func for func in domain_gen.IND2FUN.values()]

def gen_dataset(func: Callable[[], str], count: int, *, store_path: Path|None = None) -> list[str]:
    dataset = []
    for _ in range(count):
        dataset.append(func())
    if store_path:
        store_path.write_text('\n'.join(dataset))
    return dataset

def train_multi(data: Path, model_path: Path, **kwargs):
    device = kwargs.get("device", "cuda:0")
    split_size = kwargs.get('splits', 4)
    with data.open() as ds:
        dataset_table = defaultdict(list)
        while line := ds.readline():
            splits = [''.join(batch) for batch in batched(line.replace('\n',''), n=split_size)]
            for idx, split in enumerate(splits):
                dataset_table[idx].append(split or '')
        
        model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False, device=device)
        datasets = list(map(lambda d: DGADataset(d, real_domains=[]), dataset_table.values()))
        trainer = ModelTrainer(model=model, dataset=datasets[0],model_path=model_path, device=device)
        trainer.multi_train(datasets)
        return trainer

def build_regex(dataset: Path, model_path: Path, **kwargs) -> tuple[str,list[str]]:
    device = kwargs.get('device', 'cuda:0')
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False, device=device)
    model.to(device)
    graphing_path = kwargs.get('graphing_path', Path('graphs'))
    visualize = kwargs.get('visualize', False)
    regex_list = []
    max_depth = kwargs.get('max_depth', 4)
    heuristic : Heuristic = kwargs.get('heuristic', Threshold(threshold=0.6, max_depth=4))
    for num, model_path in enumerate(sorted(model_path.iterdir()), start=1):
        model.load_state_dict(torch.load(model_path, map_location=device))
        dfa = DFA(model, heuristic=heuristic)
        dfa.build_tree(max_depth=heuristic.max_depth)
        if visualize:
            dfa.visualize_tree(graphing_path/f'dfa-{num}.svg', open_file=True)
        dfa.save_file(Path(graphing_path / f'dfa-{num}.gml.gz'))
        dfa.simplify_tree()
        if visualize:
            dfa.visualize_tree(graphing_path/f'dfa-{num}-simplified.svg', open_file=True)
        regex = dfa.build_regex()
        dfa.save_file(Path(graphing_path / f'dfa-{num}-simple.gml.gz'))
        regex_list.append(regex)
    final_regex = ''.join(regex_list)
    return final_regex, regex_list

def test_regex(dataset: Iterable[str], regex:str):
        matched = 0
        total = 0
        pattern = re.compile(regex)
        for data in dataset:
            total += 1
            match = pattern.search(data)
            if match:
                print(data, match)
                matched += 1
        return matched, total

def evaluation(dataset: Path, model_path: Path, domain_name: str, **kwargs):
    threshold_num = 100
    split_sizes =  torch.tensor([2,3,4,5], dtype=torch.int8)
    thresholds = torch.linspace(0.01, 0.8, threshold_num)
    tranco_list = kwargs.get('real_domains', Path(r'data/top-1m.csv'))
    real_domains = pd.read_csv(tranco_list).to_numpy()[:,1]
    conn = sqlite3.connect('results.db')
    conn.execute('''
    PRAGMA journal_mode=WAL;
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS results(
        domain_name TEXT,
        threshold REAL,
        split_size INTEGER,
        true_positives INTEGER,
        DGA_total INTEGER,
        false_positives INTEGER,
        real_domain_total INTEGER,
        regex TEXT,
        regex_parts TEXT
    ) STRICT;
    ''')
    conn.commit()
    conn.close()
    for size in split_sizes:
        training_path = model_path / domain_name / str(int(size.item()))
        train_multi(dataset, training_path, splits=size)
        for threshold in thresholds:
            with sqlite3.connect('results.db') as conn:
                print(f"Evaluation for threshold={threshold}, split size: {size}")
                regex, regex_list = build_regex(dataset, training_path, heuristic=Threshold(threshold=threshold.item(), max_depth=int(size.item())))
                print(regex)
                print('\t'.join(regex_list))
                with dataset.open() as ds:
                    TP, total_dga = test_regex(map(lambda l: l.strip('\n'), ds.readlines()), regex)
                    FP, total_domains = test_regex(real_domains, regex)
                    conn.execute('''
                    INSERT INTO results (domain_name, threshold, split_size, true_positives, DGA_total, false_positives, real_domain_total, regex, regex_parts)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ''', (domain_name, threshold.item(), size.item(), TP, total_dga, FP, total_domains, regex, '\t'.join(regex_list)))
                    conn.commit()
if __name__ == "__main__":
    parser = ArgumentParser(description='Run this main file to generate a regular expression from a Language Model.')
    parser.add_argument("--steps", help="which additional steps should the program do in this run.", action='append', choices=['gen-dataset', 'train-models', 'test-regex', 'evaluate'])
    dataset_group = parser.add_argument_group("dataset", "options for when gen-dataset was added as extra step")
    parser.add_argument('--model-path', type=Path, help="Path where the generated models should be stored if train-models was chosen otherwise the directory that will get read for the models.", required=True)
    dataset_group.add_argument('--domain-generator', help="The generator function to use from domain_gen", choices=[func.__name__ for func in choices])
    dataset_group.add_argument('--store-dataset', action='store_true', help="Set this to specify that the dataset should be stored after generating it.")
    dataset_group.add_argument('--count', type=int, default=100000, help="The amount of entries to create when generating the data.")
    parser.add_argument('data', type=Path, help="Path to the dataset as a txt file with one entry per line, or the path to store the data at when gen-dataset is set.")
    parser.add_argument('--device', help="The device used for the model, defaults to cuda:0", default="cuda:0")
    arguments = parser.parse_args()
    dataset_gen_flag = 'gen-dataset' in arguments.steps if arguments.steps else False
    train_model_flag = 'train-models' in arguments.steps if arguments.steps else False
    test_regex_flag = 'test-regex' in arguments.steps if arguments.steps else False
    evaluate_flag = 'evaluate' in arguments.steps if arguments.steps else False
    if dataset_gen_flag and not arguments.data:
        raise Exception("Please provide the path to store the data at with --data.")
    if dataset_gen_flag and not arguments.domain_generator:
        raise Exception("Provide the generator you want to use for generating the dataset")

    logging.basicConfig()
    if evaluate_flag:
        data_path: Path = arguments.data
        if not data_path.exists():
            data_path.mkdir(exist_ok=True, parents=True)
        if not data_path.is_dir():
            raise Exception('The data path must be a directory')
        for func in choices:
            dataset_path = arguments.data / (func.__name__ + '.txt')
            gen_dataset(func, count=100_000, store_path=dataset_path)
            evaluation(dataset=dataset_path,
                       model_path=arguments.model_path,
                       domain_name=func.__name__
                       )
    else:
        if dataset_gen_flag:
            func = getattr(domain_gen, arguments.domain_generator)
            gen_dataset(func, arguments.count, store_path=arguments.data)
        if train_model_flag:
            train_multi(arguments.data, arguments.model_path, device=arguments.device)
        if test_regex_flag:
            with arguments.data.open() as f:
                lines = f.readlines()
                regex, regex_parts = build_regex(map(lambda l: l.strip('\n'), lines), arguments.model_path)
                for part in regex_parts:
                    matched, total = test_regex(list(map(lambda l: l.strip('\n'), lines)), part)
                    print(f"Success Rate: {matched/total}")
                    print(f"RegEx part: {part}")

