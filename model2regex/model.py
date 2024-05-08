#!/usr/bin/python
from torch import nn
from string import ascii_lowercase, digits


class IllegalStartChar(Exception):
    pass


class DGAClassifier(nn.Module):
    """
    A classifier using a GRU rnn model and a simple feed forward network for
    classifying DGAs
    """

    vocabular: str = ascii_lowercase + digits + "-."

    def __init__(self, emb: int, size: int, nlayers: int, **kwargs):
        """
        """
        super(DGAClassifier, self).__init__()
        start_char = kwargs.get("start_char", "_")
        if start_char in self.vocabular:
            raise IllegalStartChar("The start character should not be part of the default vocabular")
        if len(start_char) > 1:
            raise IllegalStartChar("The start character should be of length 1")
        vocabSize = len(self.vocabular) + len(start_char)
        self.embedding = nn.Embedding(num_embeddings=vocabSize,
                                      embedding_dim=emb)
        self.rnn = nn.GRU(input_size=emb, hidden_size=size,
                          num_layers=nlayers)
        self.out = nn.Linear(in_features=size, out_features=1)
        self.drop = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        x = hidden_state[-1, :]
        x = self.drop(x)
        x = self.out(x)
        x = self.sig(x)
        return x, hidden_state.detach()
