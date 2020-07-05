"""
@github{
    title = {PyTorch Transformer},
    author = {Soohwan Kim},
    pyblisher = {GitHub},
    url = {https://github.com/sooftware/PyTorch-Transformer},
    year = {2020}
}
"""
import torch
import torch.nn as nn
from nn.modules import Linear
from nn.embeddings import Embedding, PositionalEncoding
from nn.layers import TransformerEncoderLayer, TransformerDecoderLayer


class Transformer(nn.Module):
    """
    A Transformer model. User is able to modify the attributes as needed.
    The architecture is based on the paper "Attention Is All You Need".
    https://arxiv.org/abs/1706.03762

    Args: d_model, num_heads, num_layers, num_classes

    """
    def __init__(self, num_classes: int, pad_id: int,
                 num_input_embeddings: int, num_output_embeddings: int,
                 d_model: int = 512,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 num_heads: int = 8, dropout_p: float = 0.3):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_encoder_layers, num_heads, dropout_p)
        self.decoder = TransformerDecoder(d_model, num_decoder_layers, num_heads, dropout_p)
        self.linear = Linear(d_model, num_classes)

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor):
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs)
        result = self.linear(decoder_outputs)
        return result


class TransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.
    """
    def __init__(self, num_embeddings: int, d_model: int = 512, num_layers: int = 6, num_heads: int = 8,
                 dropout_p: float = 0.3, pad_id: int = 0):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_embedding = Embedding(num_embeddings, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(num_embeddings, d_model, dropout_p)
        self.layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(num_layers)])

    def forward(self):
        pass


class TransformerDecoder(nn.Module):
    """ The TransformerDecoder is composed of a stack of N identical layers. """
    def __init__(self, num_embeddings, d_model: int = 512, num_layers: int = 6, num_heads: int = 8,
                 dropout_p: float = 0.3, pad_id: int = 0):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_embedding = Embedding(num_embeddings, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(num_embeddings, d_model, dropout_p)
        self.layers = nn.ModuleList([TransformerDecoderLayer() for _ in range(num_layers)])

    def forward(self):
        pass
