import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)

class AdversarialRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh(),
        )

    def forward(self, sr):
        return torch.clamp(sr + 0.1 * self.refine_conv(sr), 0, 1)
