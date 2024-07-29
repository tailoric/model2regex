from pathlib import Path
from model2regex import *
from itertools import batched
import domain_gen.generate_domain

import re
import logging

def train_multi():
    with open('datasets/simple.txt', 'r') as simple:
        dataset_table = []
        while line := simple.readline():
            splits = [''.join(batch) for batch in batched(line[:-1], n=4)]
            if not dataset_table:
                dataset_table = [[] for _ in range(len(splits))]
            for idx, split in enumerate(splits):
                dataset_table[idx].append(split or '')
        
        model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classifying=False)
        datasets = list(map(lambda d: DGADataset(d, real_domains=[]), dataset_table))
        trainer = ModelTrainer(model=model, dataset=datasets[0],model_path=Path('models/simple'))
        trainer.multi_train(datasets)
        return trainer

def build_regex(trainer):
    model = DGAClassifier(**DEFAULT_MODEL_SETTINGS,classify=False)
    model.to('cuda:0')
    simple_models = Path('models/simple/')
    regex_list = []
    for num, model_path in enumerate(simple_models.iterdir(), start=1):
        model.load_state_dict(torch.load(model_path))
        dfa = DFA(model, store_path=Path(''), heuristic=Threshold(threshold=0.1, quantile=0.5))
        dfa.build_tree()
        regex = dfa.build_regex()
        dfa.save_file(Path(f'graph_simple_domain/dfa-{num}.gml.gz'))
        regex_list.append(regex)
        dfa.visualize_tree(f'graph_simple_domain/dfa-{num}.svg')
    final_regex = ''.join(regex_list)
    return final_regex

logging.basicConfig()
trainer = train_multi()
regex = build_regex(trainer)

dga_regex = re.compile(regex)

matched = 0
print(f"Test for regex: r'{regex}'")
total = 0
for i in range(100_000):
    line = domain_gen.generate_domain.generate_url_scheme_1()
    match = dga_regex.match(line)
    if match:
        matched += 1
    else:
        print(f"matching {line[:-1]} failed")
    total += 1

print(f"result for regex: r'{regex}'")
print(f"{matched:,}/{total:,} domains matched ({matched/total:%})")

