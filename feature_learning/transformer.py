import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.multi_head_attention = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout_attention = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, dff, dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        x = inputs
        # Put normalization layer inside residual connection according to https://arxiv.org/pdf/2002.04745.pdf
        x = self.layernorm1(x)
        attention_output = self.multi_head_attention(x, mask=mask)
        x = x + self.dropout_attention(attention_output)

        x = self.layernorm2(x)
        feedforward_output = self.feedforward(x)
        x = x + self.dropout_ff(feedforward_output)

        return x


class TokenLearner(nn.Module):
    def __init__(self, token_dim, num_selected_tokens):
        super().__init__()
        self.token_dim = token_dim
        self.num_selected_tokens = num_selected_tokens
        # Token selection network
        self.selection_net = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_selected_tokens)
        )
        self.layernorm = nn.LayerNorm(token_dim, eps=1e-6)


    def forward(self, x):
        # x has shape (batch_size, sequence length, token_dim)

        token_importance = self.selection_net(x)
        token_importance = F.softmax(token_importance, dim=1)

        # Select tokens
        selected_tokens = torch.bmm(token_importance.transpose(1, 2), x)

        return selected_tokens


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, d_hid, nlayers, d_ff, dropout=0.5, max_ep_len=600,
                 use_cnn_in_transformer=False):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(d_model)
        self.embed_sa = nn.Linear(input_size, d_model)
        self.embed_timesteps = nn.Embedding(max_ep_len, d_model)
        # self.tokenlearner = TokenLearner(token_dim=d_model, num_selected_tokens=50)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_hid, dropout) for _ in range(nlayers)])
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.use_cnn_in_transformer = use_cnn_in_transformer
        if self.use_cnn_in_transformer:
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # x has shape (batch_size, sequence length, input_size)
        # encode state and action
        x = self.embed(x)

        # encode timesteps
        timesteps = torch.arange(x.shape[1]).to(x.device)
        timesteps = self.embed_timesteps(timesteps)
        x = x + timesteps

        # if self.use_cnn_in_transformer:
        #     # Reshape to (batch_size, input_size, sequence length)
        #     x = x.transpose(1, 2)
        #     x = self.cnn_layers(x)
        #     # Reshape back to (batch_size, sequence length // 4, input_size)
        #     x = x.transpose(1, 2)
        # x = self.tokenlearner(x)
        # Add the CLS token
        # cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        # x = torch.cat((cls_token, x), dim=1)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.feedforward(x)
        return x
