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
import datetime
import time

import random
import torch
import numpy as np


choices: Sequence[Callable] = [func for func in domain_gen.IND2FUN.values()]

class StopWatch:
    def __init__(self):
        self.start_time: float|None = None
        self.recorded_timestamps = defaultdict(list)

    def start(self):
        self.start_time = time.time()

    def record_time(self, label: str):
        assert self.start_time
        self.recorded_timestamps[label].append(time.time() - self.start_time)

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

def build_regex(dataset: Path, model_path: Path, max_depth: int, check_for_uniform: bool, **kwargs) -> tuple[str,list[str]]:
    device = kwargs.get('device', 'cuda:0')
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False, device=device)
    model.to(device)
    graphing_path: Path = kwargs.get('graphing_path', Path('graphs'))
    visualize = kwargs.get('visualize', False)
    stop_watch: StopWatch|None = kwargs.get('timer')
    regex_list = []
    heuristic : Heuristic = kwargs.get('heuristic', Threshold(threshold=0.6, max_depth=max_depth))
    for num, model_path in enumerate(sorted(model_path.iterdir()), start=1):
        model.load_state_dict(torch.load(model_path, map_location=device))
        dfa = DFA(model, heuristic=heuristic)
        if stop_watch:
            stop_watch.start()
        dfa.build_tree(max_depth=max_depth)
        if stop_watch:
            stop_watch.record_time(label="tree_build_time_seconds")
        if visualize:
            graphing_path.mkdir(exist_ok=True, parents=True)
            dfa.visualize_tree(graphing_path/f'dfa-{num}.svg', open_file=True)
        dfa.save_file(Path(graphing_path / f'dfa-{num}.gml.gz'))
        if stop_watch:
            stop_watch.start()
        dfa.simplify_tree(check_for_uniform=check_for_uniform)
        if stop_watch:
            stop_watch.record_time(label="simplify_build_time_seconds")
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
            match = pattern.fullmatch(data)
            if match:
                matched += 1
        return matched, total

def evaluation(dataset: Path, model_path: Path, domain_name: str, **kwargs):
    threshold_num = 100
    split_sizes =  torch.tensor([2,3,4,5], dtype=torch.int8)
    thresholds = torch.linspace(0.01, 1.0, threshold_num)
    tranco_list = kwargs.get('real_domains', Path(r'data/top-1m.csv'))

    real_domains = pd.read_csv(tranco_list).to_numpy()[:,1]
    run_id = kwargs.get('run_id', int(time.time()))
    db_file = kwargs.get('db_file', Path(f"results.db"))
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute('''
    PRAGMA journal_mode=WAL;
    ''')
    conn.execute('''
    PRAGMA foreign_key=1;
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS result(
        result_id INTEGER PRIMARY KEY,
        domain_name TEXT,
        threshold REAL,
        split_size INTEGER,
        true_positives INTEGER,
        DGA_total INTEGER,
        false_positives INTEGER,
        real_domain_total INTEGER,
        regex TEXT,
        regex_parts TEXT,
        with_kl INTEGER,
        run_id INTEGER
    ) STRICT;
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS run_time (
        runtime_id INTEGER PRIMARY KEY ,
        result_id INTEGER UNIQUE,
        tree_build_time_seconds REAL,
        simplify_build_time_seconds REAL,
        FOREIGN KEY(result_id) REFERENCES result(result_id)
    ) STRICT;
    ''')
    conn.commit()
    conn.close()
    for size in split_sizes:
        if size >= 5 and domain_name == 'simpleExreg':
            continue
        training_path = model_path / domain_name / str(int(size.item()))
        train_multi(dataset, training_path, splits=size)
        for threshold in thresholds:
            for with_kl in (False, True):
                with sqlite3.connect(db_file) as conn:
                    conn.row_factory = sqlite3.Row
                    print(f"Evaluation for threshold={threshold}, split size: {size}")
                    stopwatch = StopWatch()
                    regex, regex_list = build_regex(dataset, training_path, heuristic=Threshold(threshold=threshold.item(), max_depth=int(size.item())), max_depth=int(size.item()),check_for_uniform=with_kl, timer=stopwatch)
                    print(regex)
                    print('\t'.join(regex_list))
                    with dataset.open() as ds:
                        TP, total_dga = test_regex(map(lambda l: l.strip('\n'), ds.readlines()), regex)
                        FP, total_domains = test_regex(real_domains, regex)
                        insert_data = {
                            'domain_name': domain_name,
                            'threshold': threshold.item(),
                            'split_size': size.item(),
                            'true_positives': TP,
                            'DGA_total': total_dga,
                            'false_positives': FP,
                            'real_domain_total': total_domains,
                            'regex': regex,
                            'regex_parts': '\t'.join(regex_list),
                            'with_kl': with_kl,
                            'run_id': run_id
                        }
                        cur = conn.execute('''
                        INSERT INTO result (domain_name, threshold, split_size, true_positives, DGA_total, false_positives, real_domain_total, regex, regex_parts, with_kl, run_id)
                        VALUES (:domain_name, :threshold, :split_size, :true_positives, :DGA_total, :false_positives, :real_domain_total, :regex, :regex_parts, :with_kl, :run_id)
                        RETURNING result_id
                        ''', insert_data)
                        row = cur.fetchone()
                        runtime_results = stopwatch.recorded_timestamps.copy()
                        for key, item in runtime_results.items():
                            runtime_results[key] = sum(item)
                        runtime_results['result_id'] = row['result_id']
                        conn.execute('''
                        INSERT INTO run_time (result_id, tree_build_time_seconds, simplify_build_time_seconds)
                        VALUES (:result_id, :tree_build_time_seconds, :simplify_build_time_seconds)
                        ''', runtime_results)
                        conn.commit()
if __name__ == "__main__":
    parser = ArgumentParser(description='Run this main file to generate a regular expression from a Language Model.')
    parser.add_argument("--steps", help="which additional steps should the program do in this run.", action='append', choices=['gen-dataset', 'train-models', 'test-regex', 'evaluate'])
    dataset_group = parser.add_argument_group("dataset", "options for when gen-dataset was added as extra step")
    parser.add_argument('--model-path', type=Path, help="Path where the generated models should be stored if train-models was chosen otherwise the directory that will get read for the models.", required=True)
    parser.add_argument('--split-size', type=int, help="The sizes of chunks the model should be split into", default=4)
    parser.add_argument('--threshold', type=float, help="The thresholding of the heuristic", default=0.4)
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
            for i in range(10):
                random.seed(0)
                torch.manual_seed(i)
                np.random.seed(i)
                run_id = int(time.time())
                db_file = Path(f"results.db")
                dataset_path = arguments.data / (func.__name__ + str(int(time.time())) + '.txt')
                gen_dataset(func, count=100_000, store_path=dataset_path)
                evaluation(dataset=dataset_path,
                           model_path=arguments.model_path,
                           domain_name=func.__name__,
                           run_id=run_id,
                           db_file=db_file,
                           )
    else:
        if dataset_gen_flag:
            func = getattr(domain_gen, arguments.domain_generator)
            gen_dataset(func, arguments.count, store_path=arguments.data)
        if train_model_flag:
            train_multi(arguments.data, arguments.model_path, device=arguments.device, splits=arguments.split_size)
        if test_regex_flag:
            with arguments.data.open() as f:
                lines = f.readlines()
                threshold = Threshold(threshold=arguments.threshold, max_depth=arguments.split_size)
                regex, regex_parts = build_regex(map(lambda l: l.strip('\n'), lines), arguments.model_path, heuristic=threshold, max_depth=arguments.split_size, visualize=True, graphing_path=Path(f'graphs') / arguments.model_path / str(arguments.threshold).replace(".", "_"), check_for_uniform=True)
                matched, total = test_regex(list(map(lambda l: l.strip('\n'), lines)), regex)
                print(f"Success Rate: {matched/total}")
                print(f"RegEx part: {regex}")
