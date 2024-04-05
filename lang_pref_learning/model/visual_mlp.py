import torch.nn as nn
import torch

from einops import rearrange


class VisualMLP(nn.Module):
    def __init__(self, in_feature_dim, action_dim, hidden_dim, out_dim, 
                 seq_len=500, n_frames=3):
        super().__init__()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(in_features=in_feature_dim, out_features=hidden_dim),
        #     nn.ReLU(),
        # )
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_feature_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim + action_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.ReLU(),
        )

        self.seq_len = seq_len
        self.n_frames = n_frames

    def forward(self, visual_features, actions):
        # import ipdb; ipdb.set_trace()
        x = rearrange(visual_features, 'b s (t d) -> (b s) d t', t=self.n_frames)
        x = self.conv1d(x)
        x = x.flatten(1)
        x = rearrange(x, '(b s) d -> b s d', s=self.seq_len)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc2(x)
        return x
