import torch.nn as nn
import torch

from einops import rearrange


class VisualMLP(nn.Module):
    def __init__(self, visual_feature_dim, state_dim, hidden_dim, out_dim, 
                 n_frames=3, state_embed_dim=128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=state_embed_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(visual_feature_dim + state_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.n_frames = n_frames

    def forward(self, inputs):
        """
        Args:
            inputs (dict): A dictionary containing the following keys:
                - img_obs (torch.Tensor): A tensor of shape (batch_size, timesteps, feature_dim)
                - state (torch.Tensor): A tensor of shape (batch_size, timesteps, state_dim)
                - actions (torch.Tensor): A tensor of shape (batch_size, timesteps, action_dim)
        
        Returns:
            A tensor of shape (batch_size, timesteps, out_dim)
        """
        visual_features = inputs['img_obs']
        state, actions = inputs['state'], inputs['actions']

        state_input = torch.cat([state, actions], dim=-1)
        state_embed = self.state_encoder(state_input)

        x = torch.cat([visual_features, state_embed], dim=-1)
        x = self.fc(x)

        return x
