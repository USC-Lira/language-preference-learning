import torch.nn as nn
import torch


class VisualMLP(nn.Module):
    def __init__(self, in_feature_dim, action_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_feature_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim + action_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.ReLU(),
        )

    def forward(self, visual_features, actions):
        x = self.fc1(visual_features)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc2(x)
        return x
