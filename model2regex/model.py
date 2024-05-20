#!/usr/bin/python
import torch
import collections
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from string import ascii_lowercase, digits
from typing import Optional

from functools import singledispatchmethod


DEFAULT_MODEL_SETTINGS = {
            "emb": 64,
            "size": 128,
            "nlayers": 1,
            "start_char": "_",
        }


class BiMapping:
    """
    A bidirectonal mapping class holding two dictionaries
    used for converting characters to their index and back
    """

    def __init__(self, characters: str):
        self._dict = {ch: i for i, ch in enumerate(characters, start=1)}
        self._reverse = {i: ch for i, ch in enumerate(characters, start=1)}

    @singledispatchmethod
    def __getitem__(self, item):
        raise NotImplementedError("item must be of type int or str")

    @__getitem__.register
    def _(self, item: str) -> int:
        return self._dict.get(item, 0)

    @__getitem__.register
    def _(self, item: int) -> str:
        return self._reverse.get(item, "<END>")


class IllegalStartChar(Exception):
    pass


class DGAClassifier(nn.Module):
    """
    A classifier using a GRU rnn model and a simple feed forward network for
    classifying DGAs
    """

    vocabulary: str = ascii_lowercase + digits + "-."

    # Domains can only be 253 characters long,
    # adding the start character the max length of our tensors is 254
    max_len: int = 254

    def __init__(self, emb: int, size: int, nlayers: int, **kwargs) -> None:
        super(DGAClassifier, self).__init__()
        self.start_char = kwargs.get("start_char", "_")
        if self.start_char in self.vocabulary:
            raise IllegalStartChar("The start character should not be part of the default vocabulary.")
        if len(self.start_char) > 1:
            raise IllegalStartChar("The start character should be of length 1.")
        self.vocab = self.vocabulary + self.start_char
        vocabSize = len(self.vocab) + 1
        self.char2idx = BiMapping(self.vocab)
        self.embedding = nn.Embedding(num_embeddings=vocabSize,
                                      embedding_dim=emb)
        self.rnn = nn.GRU(input_size=emb, hidden_size=size,
                          num_layers=nlayers)
        self.decoder = nn.Linear(size, vocabSize)
        self.out = nn.Linear(in_features=size, out_features=1)
        self.drop = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()

    def charTensor(self, char_seqs: list[str], with_padding: bool = True) -> Tensor:
        """
        turn an input sequence of characters into a tensor of indices.

        Parameter
        ---------
        char_seq:
            a list of domains to turn into the tensor of characters
        with_padding: bool
            pad the vector with 0 to the max length for domains, default True.
        """
        char_seq = map(lambda c: self.start_char + c, char_seqs)

        if with_padding:
            return torch.stack(tuple(
                F.pad(torch.tensor([self.char2idx[c] for c in domain]),
                      (0, self.max_len-len(domain))) for domain in char_seq), dim=1)
        else:
            return torch.stack(tuple(torch.tensor([self.char2idx[c] for c in domain]) for domain in char_seq), dim=1)

    def forward(self,
                input_seq: list[str] | Tensor,
                hidden_state: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """
        The forward pass of data, ideally batched.

        Parameter
        ---------
        input_seq:
            Can be a list of strings or a tensor containing the indices of a mapping
            with each entry being in a column.

        hidden_state:
            The hidden state to for the rnn layer.
        """

        if isinstance(input_seq, collections.abc.Iterable) and \
                not isinstance(input_seq, Tensor):
            input_seq = self.charTensor(input_seq)
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        decoded = self.decoder(output)
        x = hidden_state[-1, :]
        x = self.drop(x)
        x = self.out(x)
        x = self.sig(x)
        return x, decoded, hidden_state.detach()
