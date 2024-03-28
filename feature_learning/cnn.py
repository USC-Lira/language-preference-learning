import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, action_dim, hidden_dim, output_dim):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        self.repr_dim = 32 * 21 * 21 + action_dim

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs, actions):
        x = self.convnet(inputs)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    encoder = CNNEncoder(3, 4, 256, 64)
    inputs = torch.randn(32, 3, 96, 96)
    actions = torch.randn(32, 4)
    output = encoder(inputs, actions)
    print(output.shape)
    print(output)
