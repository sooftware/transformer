import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention propsed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dim)

        if mask is not None:
            score.masked_fill_(mask, -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.linear_q = nn.Linear(d_model, self.d_head * num_heads)
        self.linear_k = nn.Linear(d_model, self.d_head * num_heads)
        self.linear_v = nn.Linear(d_model, self.d_head * num_heads)
        self.linear_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = value.size(0)
        residual = query

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxTxNxD
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxTxNxD
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxTxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxTxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxTxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxTxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        output = self.linear_out(context)
        output = self.layer_norm(output + residual)  # Add & Norm (residual connection)

        return output, attn


class PositionwiseFeedForwardNet(nn.Module):
    """
    Position-wise Feed-Forward Networks proposed in "Attention Is All You Need".
    Fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Another way of describing this is as two convolutions with kernel size 1.
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout_p: float = 0.3, mode: str = 'linear'):
        super(PositionwiseFeedForwardNet, self).__init__()
        self.mode = mode.lower()
        if self.mode == 'linear':
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout_p)
            )
        elif self.mode == 'conv':
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor):
        residual = inputs  # x : BxTxD

        if self.mode == 'linear':
            output = self.feed_forward(inputs)
            output = self.layer_norm(output + residual)

        elif self.mode == 'conv':
            output = self.conv1(inputs.transpose(1, 2))
            output = self.relu(output)
            output = self.conv2(output).transpose(1, 2)
            output = self.layer_norm(output + residual)

        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))

        return output

