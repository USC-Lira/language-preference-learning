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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

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
    def __init__(self, d_model=512, num_heads=8, dff=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.norm_layer = nn.LayerNorm(d_model, eps=1e-6)
        self.multi_head_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout_attention = nn.Dropout(dropout)
        self.add_attention = nn.Add()
        self.feedforward = FeedForward(d_model, dff, dropout)

    def forward(self, inputs, mask=None):
        x = inputs
        attention_output = self.multi_head_attention(x, mask=mask)
        x = x + self.dropout_attention(attention_output)
        x = self.norm_layer(x)

        feedforward_output = self.feedforward(x)
        x = x + feedforward_output
        x = self.norm_layer(x)

        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_hid, nlayers, d_ff, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_hid, dropout) for _ in range(nlayers)])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.decoder = nn.Linear(input_size, d_model)
        self.linear = nn.Linear(d_model, input_size)

    def forward(self, x):
        x = self.decoder(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.feedforward(x)
        x = self.linear(x)
        return x
