import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.mask = None

    def forward(self, query, key, value, mask):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dim)

        if mask is not None:
            score.masked_fill_(mask, -float('inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask


class MultiHeadAttention(nn.Module):
    r"""
    Similar to standard `dot` attention but uses multiple
    attention distributions simultaneously to select relevant items.

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing the output features from the decoder.
        - **key** (batch, k_len, d_model): tensor containing features from
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = int(d_model / num_heads)
        self.d_v = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_k)
        self.linear_q = nn.Linear(d_model, self.d_k * num_heads)
        self.linear_k = nn.Linear(d_model, self.d_k * num_heads)
        self.linear_v = nn.Linear(d_model, self.d_k * num_heads)
        self.linear_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask):
        batch_size = value.size(0)
        residual = query

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k)  # BxTxNxD
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k)      # BxTxNxD
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_v)  # BxTxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_k)  # BNxTxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_k)      # BNxTxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_v)  # BNxTxD

        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_v)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_v)  # BxTxND

        output = self.linear_out(context)
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForwardNet, self).__init__()

    def forward(self):
        pass
