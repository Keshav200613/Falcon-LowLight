import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, feature_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))))

class FuzzyMembershipLayer(nn.Module):
    def __init__(self, n_features, n_clusters=5):
        super().__init__()
        self.proj = nn.Conv2d(n_features, n_clusters, 1)
        self.sigmas = nn.Parameter(torch.ones(1, n_clusters, 1, 1))

    def forward(self, x):
        z = self.proj(x)
        logits = -z.pow(2) / (2 * F.softplus(self.sigmas).pow(2) + 1e-8)
        return torch.softmax(logits, dim=1)

class SpatiallyAdaptiveANFILLE(nn.Module):
    def __init__(self, n_features=64, n_clusters=5, n_experts=4):
        super().__init__()
        self.fuzzification = FuzzyMembershipLayer(n_features, n_clusters)
        self.rule_weights = nn.Parameter(torch.randn(n_clusters, 1))
        self.refine_conv1 = nn.Conv2d(n_clusters, 32, 3, padding=1)
        self.refine_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.refine_conv3 = nn.Conv2d(16, n_experts, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        memberships = self.fuzzification(features)
        w = self.rule_weights.view(1, -1, 1, 1)
        weighted_sum = (memberships * w).sum(dim=1, keepdim=True)
        mem_sum = memberships.sum(dim=1, keepdim=True) + 1e-8
        illum_map = 1.0 - self.sigmoid(weighted_sum / mem_sum)

        rx = self.relu(self.refine_conv1(memberships))
        rx = self.relu(self.refine_conv2(rx))
        expert_memberships = torch.softmax(self.refine_conv3(rx), dim=1)

        return illum_map, memberships, expert_memberships
