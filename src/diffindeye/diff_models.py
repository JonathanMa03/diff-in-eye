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
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalUNetDDPM(nn.Module):
    def __init__(self, time_emb_dim=32, base_channels=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.time_proj1 = nn.Linear(time_emb_dim, base_channels)
        self.time_proj2 = nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj3 = nn.Linear(time_emb_dim, base_channels * 4)

        self.enc1 = ConvBlock(3, base_channels)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.down2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x_t, masked_image, mask, t):
        x = torch.cat([x_t, masked_image, mask], dim=1)

        t_emb = self.time_mlp(t.float())

        e1 = self.enc1(x)
        e1 = e1 + self.time_proj1(t_emb)[:, :, None, None]

        e2 = self.enc2(self.down1(e1))
        e2 = e2 + self.time_proj2(t_emb)[:, :, None, None]

        b = self.bottleneck(self.down2(e2))
        b = b + self.time_proj3(t_emb)[:, :, None, None]

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)