import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, action_dim, hidden_dim, output_dim):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.repr_dim = 32 * 10 * 10 + action_dim

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, imgs, actions):
        # inputs is a tensor of shape (batch_size, timesteps, n_channels, height, width)
        imgs = imgs / 255.0
        original_imgs_shape = imgs.shape
        if len(original_imgs_shape) == 5:
            imgs = imgs.view(-1, *imgs.shape[-3:])

        if len(actions.shape) == 3:
            actions = actions.view(-1, actions.shape[-1])

        x = self.convnet(imgs)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, actions], dim=-1)
        x = self.fc(x)

        if len(original_imgs_shape) == 5:
            # Reshape back to (batch_size, timesteps, output_dim)
            x = x.view(original_imgs_shape[0], -1, x.shape[-1])
        return x


if __name__ == "__main__":
    # Test the CNN encoder
    encoder = CNNEncoder(in_channels=3, action_dim=4, hidden_dim=256, output_dim=128)
    imgs = torch.randn(32, 10, 3, 96, 96)
    actions = torch.randn(32, 10, 4)
    output = encoder(imgs, actions)
    print(output.shape)
