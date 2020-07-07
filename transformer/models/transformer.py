"""
Author:
    - **Soohwan Kim @sooftware**
    - **Email: sh951011@gmail.com**

Reference :
    - **https://github.com/graykode/nlp-tutorial**
    - **https://github.com/dreamgonfly/transformer-pytorch**
    - **https://github.com/jadore801120/attention-is-all-you-need-pytorch**
    - **https://github.com/JayParks/transformer**
"""
import torch
import torch.nn as nn
from transformer.models.modules import Linear
from transformer.models.mask import subsequent_masking, pad_masking
from transformer.models.embeddings import Embedding, PositionalEncoding
from transformer.models.layers import TransformerEncoderLayer, TransformerDecoderLayer
from typing import Optional


class Transformer(nn.Module):
    """
    A Transformer model. User is able to modify the attributes as needed.
    The architecture is based on the paper "Attention Is All You Need".

    Args:
        num_classes (int): the number of classfication
        pad_id (int): identification of <PAD_token>
        num_input_embeddings (int): dimension of input embeddings
        num_output_embeddings (int): dimenstion of output embeddings
        d_model (int): dimension of model (default: 512)
        d_ff (int): dimension of feed forward network (default: 2048)
        num_encoder_layers (int): number of encoder layers (default: 6)
        num_decoder_layers (int): number of decoder layers (default: 6)
        num_heads (int): number of attention heads (default: 8)
        dropout_p (float): dropout probability (default: 0.3)
        ffnet_style (str): if poswise_ffnet is 'ff', position-wise feed forware network to be a feed forward,
            otherwise, position-wise feed forward network to be a convolution layer. (default: ff)

    Inputs: inputs, targets
        - **inputs** (batch, input_length): tensor containing input sequences
        - **targets** (batch, target_length): tensor contatining target sequences

    Returns: output
        - **output**: tensor containing the outputs
    """
    def __init__(self, num_classes: int, pad_id: int, num_input_embeddings: int, num_output_embeddings: int,
                 d_model: int = 512, d_ff: int = 2048, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dropout_p: float = 0.3, ffnet_style: str = 'ff'):
        super(Transformer, self).__init__()
        self.pad_id = pad_id
        self.encoder = TransformerEncoder(
            num_embeddings=num_input_embeddings,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=dropout_p,
            pad_id=pad_id
        )
        self.decoder = TransformerDecoder(
            num_embeddings=num_output_embeddings,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id
        )
        self.linear = Linear(d_model, num_classes)

    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor]):
        input_length, target_length = inputs.size(1), targets.size(1)

        inputs_mask = pad_masking(inputs, input_length, self.pad_id)
        memory_mask = pad_masking(inputs, target_length, self.pad_id)
        targets_mask = subsequent_masking(targets)

        memory, encoder_self_attns = self.encoder(inputs, inputs_mask)
        output, decoder_self_attns, decoder_encoder_attns = self.decoder(targets, memory, targets_mask, memory_mask)
        output = self.linear(output)

        return output, encoder_self_attns, decoder_self_attns, decoder_encoder_attns


class TransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.
    """
    def __init__(self, num_embeddings: int, d_model: int = 512, d_ff: int = 2048,
                 num_layers: int = 6, num_heads: int = 8, ffnet_style: str = 'ff',
                 dropout_p: float = 0.3, pad_id: int = 0):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.embedding = Embedding(num_embeddings, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor):
        self_attns = list()
        output = None

        inputs = self.embedding(inputs)
        inputs = self.positional_encoding(inputs)

        for layer in self.layers:
            output, attn = layer(inputs, inputs_mask)
            self_attns.append(attn)
            inputs = output

        return output, self_attns


class TransformerDecoder(nn.Module):
    """
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.
    """
    def __init__(self, num_embeddings, d_model: int = 512, d_ff: int = 512,
                 num_layers: int = 6, num_heads: int = 8, ffnet_style: str = 'ff',
                 dropout_p: float = 0.3, pad_id: int = 0):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(num_embeddings, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ff,  dropout_p, ffnet_style) for _ in range(num_layers)]
        )

    def forward(self, inputs, memory, inputs_mask, memory_mask):
        self_attns, encoder_attns = list(), list()
        output = None

        inputs = self.embedding(inputs)
        inputs = self.positional_encoding(inputs)

        for layer in self.layers:
            output, self_attn, encoder_attn = layer(inputs, memory, inputs_mask, memory_mask)
            self_attns.append(self_attn)
            encoder_attns.append(encoder_attn)
            inputs = output

        return output, self_attns, encoder_attns
