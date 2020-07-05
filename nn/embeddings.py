import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.3, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, embedded: torch.Tensor, step: Optional[int] = None):
        if step is None:
            embedded += self.pe[:, :embedded.size(1)]
        else:
            embedded += self.pe[:, step]
            
        return self.dropout(embedded)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, pad_id: int, d_model):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: torch.Tensor):
        embedded = self.embedding(inputs)
        embedded *= self.sqrt_dim
        return embedded
