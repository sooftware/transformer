import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, num_embeddings: int, d_model: int = 512, dropout: float = 0.3):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = self.get_sinusoid_encoding_table(num_embeddings, d_model)

    def forward(self, embedded: torch.Tensor, step: Optional[int] = None):
        if step is None:
            embedded += self.pe[:, :embedded.size(1)]
        else:
            embedded += self.pe[:, step]

        return self.dropout(embedded)

    def get_sinusoid_encoding_table(self, num_embeddings: int, d_model: int = 512):
        def cal_angle(pos: int, i: int):
            return pos / np.power(10000, 2 * (i // 2) / d_model)  # i // 2: (2i, 2i +1) => 2i

        def get_pos_angle_vector(pos: int):
            return [cal_angle(pos, i) for i in range(d_model)]

        sinusoid_table = np.array([get_pos_angle_vector(pos) for pos in range(num_embeddings)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, pad_id: int, d_model):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: torch.Tensor):
        return self.embedding(inputs) * self.sqrt_dim
