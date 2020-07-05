import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()

    def forward(self):
        pass


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    """
    def __init__(self):
        super(TransformerDecoderLayer, self).__init__()

    def forward(self):
        pass
