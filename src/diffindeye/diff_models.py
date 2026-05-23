import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class ConditionalTimeDDPM(nn.Module):
    def __init__(self, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.deconv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.deconv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.relu = nn.ReLU()
        self.time_proj = nn.Linear(time_emb_dim, 64)

    def forward(self, x_t, masked_image, mask, t):
        x = torch.cat([x_t, masked_image, mask], dim=1)

        t_emb = self.time_mlp(t.float())

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))

        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.relu(self.deconv1(h))
        h = self.relu(self.deconv2(h))

        return self.deconv3(h)