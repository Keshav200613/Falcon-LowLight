import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True)
        self.feat = nn.Sequential(*list(vgg.features)[:26]).eval()
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.feat(pred), self.feat(target))

class ICMoELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.perceptual = VGGPerceptualLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_hr, target_hr, illum_map, expert_memberships, discriminator=None):
        recon_loss = torch.sqrt((pred_hr - target_hr) ** 2 + 1e-6).mean()
        percep_loss = self.perceptual(pred_hr, target_hr)

        gray = target_hr.mean(dim=1, keepdim=True)
        if gray.shape[2:] != illum_map.shape[2:]:
            gray = F.interpolate(gray, size=illum_map.shape[2:], mode='bilinear', align_corners=False)
        g_min = gray.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        g_max = gray.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        gray_n = (gray - g_min) / (g_max - g_min + 1e-8)
        illum_loss = F.l1_loss(illum_map, gray_n)
        dx = (illum_map[:, :, :, :-1] - illum_map[:, :, :, 1:]).abs()
        dy = (illum_map[:, :, :-1, :] - illum_map[:, :, 1:, :]).abs()
        illum_loss = illum_loss + 0.01 * (dx.mean() + dy.mean())

        K = self.cfg['n_experts']
        u_bar = expert_memberships.mean(dim=[0, 2, 3])
        util = self.mse(u_bar, torch.full_like(u_bar, 1.0 / K))
        ent = -(expert_memberships * torch.log(expert_memberships + 1e-8)).sum(dim=1).mean()

        adv = torch.tensor(0.0, device=pred_hr.device)
        if discriminator is not None:
            adv = self.bce(discriminator(pred_hr), torch.ones_like(discriminator(pred_hr)))

        total = (self.cfg['lambda_recon'] * recon_loss +
                 self.cfg['lambda_percep'] * percep_loss +
                 self.cfg['lambda_illum'] * illum_loss +
                 self.cfg['lambda_util'] * util +
                 self.cfg['lambda_ent'] * ent +
                 self.cfg['lambda_adv'] * adv)

        return {'total': total, 'recon': recon_loss, 'percep': percep_loss,
                'illum': illum_loss, 'util': util, 'ent': ent, 'adv': adv}
