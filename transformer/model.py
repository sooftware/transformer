"""
@github{
    title = {PyTorch Transformer},
    author = {Soohwan Kim},
    pyblisher = {GitHub},
    url = {https://github.com/sooftware/PyTorch-Transformer},
    year = {2020}
}
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    PyTorch Implementation of Transformer proposed in "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, num_heads, num_layers, num_classes):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.linear_out = nn.Linear(d_model, num_classes)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs)
        result = self.linear_out(decoder_outputs)
        return result


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self):
        pass


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention propsed in "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762

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
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, key, value, mask):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dim)

        if mask is not None:
            score.masked_fill_(mask, -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Similar to standard `dot` attention but uses multiple attention distributions simultaneously to select relevant items.
    https://arxiv.org/abs/1706.03762

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
    def __init__(self, d_model=512, num_heads=8):
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

    def forward(self, query, key, value, mask):
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
    def __init__(self):
        super(PositionwiseFeedForwardNet, self).__init__()

    def forward(self):
        pass
