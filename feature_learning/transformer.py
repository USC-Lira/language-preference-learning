import torch
import torch.nn as nn
import torch
import numpy as np
import math
import time
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch import Tensor

torch.manual_seed(0)
np.random.seed(0)

class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # TODO: Add in the multi-head attention layer and feed forward layer here!!!
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Decoder for initial mapping
        self.decoder = nn.Linear(input_size, d_model)

        # Transformer input size = d_model b/c we have embeddings
        self.linear = nn.Linear(d_model, input_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # TODO: Add in the multi-head attention layer and feed forward layer here!!!
        x = self.decoder(x)  # Assuming x is a 2D tensor of shape (sequence_length, input_size)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Change shape to (batch_size, sequence_length, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change shape back to (sequence_length, batch_size, d_model)
        x = self.linear(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MultiHeadAttention(nn.module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_hid, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x