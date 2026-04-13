import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .it2frab import IT2FuzzyResidualAttentionBlock

class FuzzyDictionaryBank(nn.Module):
    def __init__(self, n_atoms=64, atom_dim=64, n_components=4):
        super().__init__()
        self.component_atoms = nn.Parameter(torch.randn(n_components, n_atoms, atom_dim) * 0.02)
        self.illum_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, n_components)
        )
        self.query_proj = nn.Conv2d(atom_dim, n_atoms, 1)
        self.sigma = nn.Parameter(torch.ones(1, n_atoms, 1, 1))

    def forward(self, features, illumination_map):
        illum_scalar = illumination_map.mean(dim=[2, 3])
        comp_w = torch.softmax(self.illum_mlp(illum_scalar), dim=1)
        blended = torch.einsum('bc,cka->bka', comp_w, self.component_atoms)

        z = self.query_proj(features)
        membership = torch.softmax(-z.pow(2) / (2 * F.softplus(self.sigma).pow(2) + 1e-8), dim=1)
        return torch.einsum('bkhw,bkc->bchw', membership, blended)

class DictionaryBasedSR(nn.Module):
    def __init__(self, scale_factor=4, in_channels=64, n_atoms=64, n_components=4):
        super().__init__()
        self.dict_bank = FuzzyDictionaryBank(n_atoms, in_channels, n_components)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        n_stages = int(math.log2(scale_factor))
        layers = []
        for _ in range(n_stages):
            layers += [
                nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)]
        self.upscale = nn.Sequential(*layers)
        self.hr_refinement = IT2FuzzyResidualAttentionBlock(n_feat=in_channels, n_rules=4)
        self.to_rgb = nn.Sequential(nn.Conv2d(in_channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, features, illumination_map):
        prior = self.dict_bank(features, illumination_map)
        fused = self.fusion_conv(torch.cat([features, prior], dim=1))
        hr_feat = self.upscale(fused)
        refined = self.hr_refinement(hr_feat)
        return self.to_rgb(refined)
