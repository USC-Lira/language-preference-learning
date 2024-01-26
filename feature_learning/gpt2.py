import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CasualAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_ep_len, dropout=0.0, scale=True):
        super(CasualAttention, self).__init__()
        self.n_embed = d_model
        self.n_head = num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_ep_len, max_ep_len), dtype=torch.bool)).view(
                1, 1, max_ep_len, max_ep_len
            )
        )

        self.scale = scale

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k.transpose(-2, -1))
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        nd, ns = w.size(-2), w.size(-1)
        casual_mask = self.bias[:, :, ns - nd: ns, :ns]
        mask_value = torch.finfo(w.dtype).min
        w = torch.where(casual_mask, w, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = F.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        q, k, v = x, x, x
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn_outputs = self._attn(q, k, v, attention_mask=mask)
        return attn_outputs


# Test
casual_attn = CasualAttention(d_model=128, num_heads=4, max_ep_len=64, dropout=0.1, scale=True)
x = torch.randn(1, 64, 128)
outputs = casual_attn(x)
