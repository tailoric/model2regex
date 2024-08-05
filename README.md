# model2regex

A python library for generating RegEx from a language model.

## Setup

- Install python 3.12 
- Install [`poetry`](https://python-poetry.org/docs/#installing-with-pipx)
- run `poetry install` 

## Training

- To start training use `poetry run python -m model2regex`
- To run all the unittests use `poetry run python -m unittest discover -s model2regex -p "*_test.py"`

## Main file

- [`main.py`](main.py) will run the current possible steps, use `poetry run python main.py -h` for a help message
  - Example call `poetry run python main.py --steps gen-dataset --steps train-models --model-path model_abc --domain-generator generate_url_scheme_1 abc-domains.txt`
