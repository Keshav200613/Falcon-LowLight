import torch
import torch.nn as nn
import torch.nn.functional as F

class IT2FuzzyMembershipAttention(nn.Module):
    def __init__(self, n_feat, n_rules=4):
        super().__init__()
        self.rule_proj = nn.Conv2d(n_feat, n_rules, 1)
        self.log_sigma_lower = nn.Parameter(torch.zeros(1, n_rules, 1, 1))
        self.log_sigma_upper = nn.Parameter(torch.full((1, n_rules, 1, 1), 0.5))
        self.attention_proj = nn.Sequential(nn.Conv2d(n_rules, n_feat, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.rule_proj(x)
        s_lower = F.softplus(self.log_sigma_lower) + 1e-6
        s_upper = F.softplus(self.log_sigma_upper) + s_lower
        mu_l = torch.exp(-z.pow(2) / (2 * s_lower.pow(2)))
        mu_u = torch.exp(-z.pow(2) / (2 * s_upper.pow(2)))
        firing = (mu_u + mu_l) / 2
        return self.attention_proj(firing)

class IT2FuzzyResidualAttentionBlock(nn.Module):
    def __init__(self, n_feat, n_rules=4):
        super().__init__()
        self.struct_branch = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 5, padding=2, groups=n_feat),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
        )
        self.fuzzy_attn = IT2FuzzyMembershipAttention(n_feat, n_rules=n_rules)
        self.fuse = nn.Conv2d(n_feat * 2, n_feat, 1)

    def forward(self, x):
        struct = self.struct_branch(x)
        detail = self.detail_branch(x)
        attn = self.fuzzy_attn(x)
        blended = self.fuse(torch.cat([attn * struct, (1 - attn) * detail], dim=1))
        return x + blended
