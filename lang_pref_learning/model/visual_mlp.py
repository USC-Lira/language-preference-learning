import torch.nn as nn
import torch

from einops import rearrange


class VisualMLP(nn.Module):
    def __init__(self, in_feature_dim, action_dim, hidden_dim, out_dim, 
                 n_frames=3):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_feature_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        # self.conv2d = nn.Sequential(
        #     nn.Conv2d(in_channels=in_feature_dim, out_channels=256, kernel_size=3),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )
        self.fc2 = nn.Sequential(
            # nn.Linear(in_features=hidden_dim + action_dim, out_features=hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.ReLU(),
        )
        self.n_frames = n_frames

    def forward(self, visual_features, actions):
        # seq_len = visual_features.shape[1]
        # x = rearrange(visual_features, 'b t c h w -> (b t) c h w', t=seq_len)
        # x = self.conv2d(x)
        # x = torch.flatten(x, 1)
        # x = self.conv1d(x)
        # x = x.flatten(1)
        # x = rearrange(x, '(b t) d -> b t d', t=seq_len)
        # x = torch.cat([x, actions], dim=-1)
        x = self.fc1(visual_features)
        x = self.fc2(x)
        return x
