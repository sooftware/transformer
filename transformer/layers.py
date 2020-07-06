import torch.nn as nn
from transformer.sublayers import MultiHeadAttention, PoswiseFeedForwardNet, AddNorm


class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout_p: float = 0.3, mode: str = 'ff'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, mode), d_model)

    def forward(self, inputs, inputs_mask):
        output, attn = self.self_attention(inputs, inputs, inputs, inputs_mask)
        output = self.feed_forward(output)
        return output, attn


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048,  dropout_p: float = 0.3, mode: str = 'ff'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.encoder_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, mode), d_model)

    def forward(self, inputs, memory, inputs_mask, memory_mask):
        output, self_attn = self.self_attention(inputs, inputs, inputs, inputs_mask)
        output, encoder_attn = self.encoder_attention(output, memory, memory, memory_mask)
        output = self.feed_forward(output)
        return output, self_attn, encoder_attn
