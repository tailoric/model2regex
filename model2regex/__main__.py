import argparse
import pathlib
import logging
import pandas as pd
from . import (
        model,
        dga,
        trainer
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model to turn into a Regular Expression.")
    parser.add_argument("--data",
                        dest='data',
                        type=pathlib.Path,
                        help="The data file with real domains",
                        default=pathlib.Path('data/top-1m.csv'))
    parser.add_argument("--out-dir",
                        dest='outdir',
                        type=pathlib.Path,
                        help="The directory where to put the trained models to",
                        default=pathlib.Path("models"))
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--size",
                        type=int,
                        help="set the size of the dataset",
                        default=2_000_000)
    parser.add_argument("--seed",
                        type=str,
                        help="the seed of the dga generation algorithm",
                        default="earnestnessbiophysicalohax.com")
    parser.add_argument("--device",
                        type=str,
                        help="the gpu or cpu device to run the model on",
                        default="cuda:0")

    args = parser.parse_args()
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level)
    domains = pd.read_csv(args.data, header=None).values[:, 1]
    dataset = dga.generate_dataset(dga.banjori, size=args.size, seed=args.seed, real_domains=domains)
    logging.debug("generated dataset of size: %d", len(dataset))
    model = model.DGAClassifier(**model.DEFAULT_MODEL_SETTINGS)
    logging.debug("Model initialized: %s", model)
    trainer = trainer.ModelTrainer(dataset=dataset, model=model,
                                   log_level=log_level,
                                   device=args.device,
                                   model_path=args.outdir,
                                   )
    logging.debug("Starting training...")
    trainer.train()
