import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvNeXtLiteExpertLayer

class RecursiveResidualGroup(nn.Module):
    def __init__(self, n_feat, n_experts=4, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvNeXtLiteExpertLayer(n_feat, num_experts=n_experts) for _ in range(n_blocks)]
        )
        self.tail = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False)

    def forward(self, x, expert_memberships):
        res = x
        for block in self.blocks:
            x = block(x, expert_memberships)
        return res + self.tail(x)

class ConvNeXtExpertRestoration(nn.Module):
    def __init__(self, n_feat=64, n_rrg=3, n_experts=4):
        super().__init__()
        self.illum_proj = nn.Sequential(nn.Conv2d(1, n_feat, 1), nn.Sigmoid())
        self.rrgs = nn.ModuleList(
            [RecursiveResidualGroup(n_feat, n_experts=n_experts) for _ in range(n_rrg)]
        )
        self.tail = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False)

    def forward(self, features, illum_map, expert_memberships):
        if illum_map.shape[2:] != features.shape[2:]:
            illum_map = F.interpolate(illum_map, size=features.shape[2:], mode='bilinear', align_corners=False)
        if expert_memberships.shape[2:] != features.shape[2:]:
            expert_memberships = F.interpolate(expert_memberships, size=features.shape[2:], mode='bilinear', align_corners=False)
        x = features * self.illum_proj(illum_map)
        for rrg in self.rrgs:
            x = rrg(x, expert_memberships)
        return features + self.tail(x)
