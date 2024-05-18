from typing import Optional
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from torch.distributions import Categorical
from pathlib import Path
from sklearn.model_selection import KFold
from copy import deepcopy
from .model import DGAClassifier, DEFAULT_MODEL_SETTINGS
from .dga import generate_dataset, banjori
import torch.nn.functional as F
import logging
import random
import torch
import sys
import argparse

class ModelTrainer:
    """
    test

    Parameter:
    ----------
    dataset: torch.utils.data.Dataset
        The dataset to use.

    model: model.DGAClassifier
        The model to use

    model_path: pathlib.Path | str
        The path to store the models in defaults to a models folder in the cwd

    tensorboard_runs: pathlib.Path | str
        The path to store the runs for visualization in tensorboard

    device: str
        The device to store models and tensors to, defaults to cuda:0

    criterion:
        A loss criterion from toch.nn, CrossEntropyLoss by default

    optimizer:
        The optimizer for the backpropagation, defaults to optim.Adam

    log_level:
        The log level for the prints defaults to logging.INFO
    """

    def __init__(self, dataset: Dataset, model: DGAClassifier, **kwargs):
        self.dataset = dataset
        self.model = model
        self.models_path = Path(kwargs.get("model_path", './models/'))
        if not self.models_path.exists():
            self.models_path.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = Path(kwargs.get("tensorboard_runs", "./runs"))

        self.writer: Optional[SummaryWriter] = None
        if self.tensorboard_dir and not self.tensorboard_dir.exists():
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self.device = kwargs.get("device", "cuda:0")
        self.criterion = kwargs.get("criterion", nn.CrossEntropyLoss(reduction="sum"))
        self.optimizer = kwargs.get("optimizer", optim.Adam(self.model.parameters(),
                                                            lr=kwargs.get("optim_lr", 0.001)))
        self.log = logging.getLogger(__name__)
        self.log.setLevel(kwargs.get("log_level", logging.INFO))
        self.model.to(self.device)

    def train(self, *, folds=5, epochs=3, save_model=True):
        """
        Train the model with k-fold cross validation.

        Parameters:
        -----------

        folds: int
            The k for k-fold cross validation, defaults to 5.

        epochs: int
            The amount of epochs to go through, defaults to 3.

        save_model: bool
            Whether the models created during cross validation should be saved to the models_path,
            defaults to True.
        """
        kfold = KFold(n_splits=folds, shuffle=True)
        untrained_model = deepcopy(self.model.state_dict())
        accuracies = []
        for fold, (dataset_train, dataset_validate) in enumerate(kfold.split(self.dataset), start=1):
            self.writer = SummaryWriter(self.tensorboard_dir / f'Classifier-prototype-fold-{fold}')
            self.model.load_state_dict(state_dict=untrained_model)
            loader_train = DataLoader(dataset=self.dataset,
                                      batch_size=500,
                                      sampler=SubsetRandomSampler(dataset_train))
            loader_validation = DataLoader(dataset=self.dataset,
                                           batch_size=500,
                                           sampler=SubsetRandomSampler(dataset_validate))
            self.train_fold(loader=loader_train, epochs=epochs)
            if save_model:
                torch.save(self.model.state_dict(), self.models_path / f'model-fold-{fold}.pth')
            total, correct = self.validate_fold(loader=loader_validation)
            accuracy = correct / total
            self.log.info("validation of fold %d: %d/%d", fold, correct, total)
            accuracies.append(accuracy)
            if self.writer:
                self.writer.close()
        for idx, accuracy in enumerate(accuracies, start=1):
            self.log.info(f"accuracy of fold {idx}: {accuracy:%}")

    def predict_next_token(self, starter: str):
        char_t = self.model.charTensor([starter])
        output, tokens, _ = self.model(char_t.permute(1, 0).squeeze().to(self.device), None)
        tokens = F.softmax(torch.squeeze(tokens[-1, :]), dim=0)
        dist = Categorical(output)
        index = dist.sample()
        return index.item()

    def predict(self, starter: str):
        for _ in range(254):
            ind = self.predict_next_token(starter)
            if ind == 0:
                starter += '<END>'
                break
            starter += self.model.char2idx[ind]
        return starter

    def train_fold(self, loader: DataLoader, epochs=3):
        self.model.train()
        total_batches = len(loader.sampler) // loader.batch_size
        for epoch in range(epochs):
            e_loss = 0
            self.log.info("epoch: %s", epoch)
            self.log.info("----------------------------------------------")
            for batch, (x, y) in enumerate(loader):
                self.optimizer.zero_grad()
                input_ = self.model.charTensor(x).to(self.device)
                output, tokens, _ = self.model(input_, None)
                shifted_input = torch.vstack((input_[1:], torch.zeros(1, loader.batch_size).to(self.device))).long()
                loss_lm = self.criterion(tokens.permute(1, 2, 0), shifted_input.permute(1, 0))
                loss_class = self.criterion(output.squeeze(), y.to(self.device))
                loss = loss_lm + loss_class
                loss.backward()
                e_loss += loss.item()
                self.optimizer.step()
                if batch in (0, total_batches // 2, total_batches-1):
                    idx = random.randint(0, len(x)-1)
                    correct = (output.permute(1, 0).round() == y.to(self.device)).sum().item()
                    self.log.info("showing one prediction of batch: %d", batch)
                    self.log.info("inputstr: %s", x[idx])
                    self.log.info("label: %d", y[idx])
                    self.log.info("output: %d", output[idx].round().item())
                    self.log.info("accuracy of batch %d", batch)
                    self.log.info("%d/%d correct.", correct, loader.batch_size)
                    self.model.eval()
                    self.log.info("prediction: %s", self.predict("_"))
                    self.model.train()
                    self.log.info("--------------------------------------------")

            self.model.train()
            self.log.info("the sum loss of epoch %d is: %f\n\n", epoch, e_loss)
            if self.writer:
                self.writer.add_scalar('Loss/epoch/train', e_loss, epoch)
        if self.writer:
            self.writer.flush()

    def validate_fold(self, *, loader: DataLoader):
        total, correct = 0, 0
        with torch.no_grad():
            self.model.eval()
            for batch, (x, y) in enumerate(loader):
                input_ = self.model.charTensor(x).to(self.device)
                output, _, _ = self.model(input_, None)
                total += y.size(0)
                correct += (output.permute(1, 0).round() == y.to(self.device)).sum().item()
        return total, correct


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    model: DGAClassifier = DGAClassifier(**DEFAULT_MODEL_SETTINGS)
    dataset = generate_dataset(banjori, 'earnestnessbiophysicalohax.com')
    trainer = ModelTrainer(model=model, dataset=dataset)
    trainer.train()
