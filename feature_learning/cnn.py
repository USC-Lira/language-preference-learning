import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, action_dim, hidden_dim, output_dim):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
        )

        self.repr_dim = 16 * 19 * 19 + action_dim

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs, actions):
        # inputs is a tensor of shape (batch_size, timesteps, n_channels, height, width)
        original_input_shape = inputs.shape
        if len(original_input_shape) == 5:
            inputs = inputs.view(-1, *inputs.shape[-3:])

        if len(actions.shape) == 3:
            actions = actions.view(-1, actions.shape[-1])

        x = self.convnet(inputs)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc(x)

        if len(original_input_shape) == 5:
            # Reshape back to (batch_size, timesteps, output_dim)
            x = x.view(original_input_shape[0], -1, x.shape[-1])
        return x
