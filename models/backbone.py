import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

    def forward(self, x):
        res = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return res + x.permute(0, 3, 1, 2)

class ConvNeXtLiteExpertLayer(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([ConvNeXtLiteBlock(dim) for _ in range(num_experts)])

    def forward(self, x, weights):
        out = torch.stack([e(x) for e in self.experts], dim=1)
        return (weights.unsqueeze(2) * out).sum(dim=1)
