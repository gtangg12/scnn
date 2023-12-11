import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

    
class ResidualBlock(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int, layers: int, stride: int, padding_mode='zeros'):
        super().__init__()
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, padding_mode=padding_mode)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.PReLU())
        for _ in range(1, layers):
            self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.PReLU())
    
    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class ResidualEncoder(nn.Module):
    """
    """
    def __init__(self, in_channels=64, latent_channels=1024, hidden_channels=256, num_blocks=6, layers_per_block=8):
        """
        """
        super().__init__()
        self.conv_in = nn.Conv2d(1, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            out_channels = min(in_channels * (2 ** i), hidden_channels)
            stride = 1 if i == 0 else 2
            block = ResidualBlock(in_channels, out_channels, layers_per_block, stride)
            self.blocks.append(block)
            in_channels = out_channels

        self.conv_out = nn.Conv2d(in_channels, latent_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.mlp = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(latent_channels, latent_channels),
            nn.PReLU(),
            nn.Linear(latent_channels, latent_channels),
            nn.PReLU(),
        ])

    def forward(self, x):
        x = self.relu(self.bn(self.conv_in(x)))
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        x = self.mlp(x)
        return x


class ResidualDecoder(nn.Module):
    """
    """
    def __init__(self, latent_channels=1024, out_channels=32, hidden_channels=256, num_blocks=3, layers_per_block=6):
        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Linear(latent_channels, latent_channels),
            nn.PReLU(),
            nn.Linear(latent_channels, latent_channels),
            nn.PReLU(),
        ])
        # first conv transpose 1024x1x1 -> 256x4x4
        self.conv_transpose_in = nn.ConvTranspose2d(latent_channels, hidden_channels, kernel_size=4, stride=4, padding=0)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # 256x4x4 -> 128x8x8 -> 64x16x16 -> 32x32x32
            in_channels  = min(hidden_channels // 1, hidden_channels // (2 ** (i - 0)))
            out_channels = min(hidden_channels // 2, hidden_channels // (2 ** (i + 1)))
            block = ResidualBlock(in_channels, out_channels, layers_per_block, stride=1)
            conv_transpose_up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.blocks.append(nn.Sequential(*[
                block,
                conv_transpose_up,
            ]))

    def forward(self, x):
        x = self.mlp(x).view(x.size(0), -1, 1, 1)
        x = self.conv_transpose_in(x)
        for block in self.blocks:
            x = block(x)
        return x


class EncoderDecoder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.encoder = ResidualEncoder()
        self.decoder = ResidualDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class EncoderClassifier(nn.Module):
    """
    """
    def __init__(self, nclasses):
        super().__init__()
        self.encoder = ResidualEncoder()
        self.classifier = nn.Sequential(*[
            nn.Linear(1024, 2048),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.PReLU(),
            nn.Linear(512, nclasses),
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    model = ResidualBlock(64, 64, 2, 1).to('cuda')
    x = torch.randn(1, 64, 128, 128).to('cuda')
    y = model(x)
    print(y.shape)

    model = ResidualEncoder().to('cuda')
    x = torch.randn(10, 1, 128, 128).to('cuda')
    y = model(x)
    print(y.shape)

    model = ResidualDecoder().to('cuda')
    x = torch.randn(10, 1024).to('cuda')
    y = model(x)
    print(y.shape)

    model = EncoderDecoder().to('cuda')
    x = torch.randn(10, 1, 128, 128).to('cuda')
    y = model(x)
    print(y.shape)