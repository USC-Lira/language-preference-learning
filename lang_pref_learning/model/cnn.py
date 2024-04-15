import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.state_embed_layer = nn.Linear(state_dim, 128)
        self.repr_dim = 64 * 5 * 5 + 128

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(192),
            transforms.Resize((112, 112), antialias=True),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((112, 112), antialias=True),
        ])

    def forward(self, inputs, train=False):
        # inputs is a tensor of shape (batch_size, timesteps, n_channels, height, width)
        imgs = inputs['img_obs']
        state, actions = inputs['state'], inputs['actions']

        original_imgs_shape = imgs.shape
        if len(original_imgs_shape) == 5:
            imgs = imgs.view(-1, *imgs.shape[-3:])

        state = rearrange(state, 'b t d -> (b t) d')
        actions = rearrange(actions, 'b t a -> (b t) a')

        # Apply the appropriate transformation
        if train:
            imgs = self.train_transform(imgs)
        else:
            imgs = self.test_transform(imgs)

        x = self.convnet(imgs)
        x = x.reshape(x.size(0), -1)

        state_embed = self.state_embed_layer(torch.cat([state, actions], dim=-1))
        x = torch.cat([x, state_embed], dim=-1)
        x = self.fc(x)

        if len(original_imgs_shape) == 5:
            # Reshape back to (batch_size, timesteps, output_dim)
            x = x.view(original_imgs_shape[0], -1, x.shape[-1])
        return x
