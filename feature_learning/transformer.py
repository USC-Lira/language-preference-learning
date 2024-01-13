import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        return self.attn(x, x, x, attn_mask=mask)[0]


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dff=2048, dropout=0.0, norm_layer=True):
        super(EncoderLayer, self).__init__()
        self.norm_layer = nn.LayerNorm(d_model, eps=1e-6)
        self.multi_head_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout_attention = nn.Dropout(dropout)
        self.feedforward = FeedForward(d_model, dff, dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        x = inputs
        # Put normalization layer inside residual connection according to https://arxiv.org/pdf/2002.04745.pdf
        attention_output = self.multi_head_attention(self.norm_layer(x), mask=mask)
        x = x + self.dropout_attention(attention_output)

        feedforward_output = self.feedforward(self.norm_layer(x))
        x = x + self.dropout_ff(feedforward_output)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_hid, nlayers, d_ff, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_hid, dropout) for _ in range(nlayers)])
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.embed = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.linear = nn.Linear(d_model, input_size)

    def forward(self, x):
        # x has shape (batch_size, sequence length, input_size)
        x = self.embed(x)
        # Add the CLS token
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.feedforward(x)
        # x = self.linear(x)
        return x
