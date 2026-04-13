import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity  as _ssim


def _to_np(t):
    return t.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()


def calculate_psnr(pred, target):
    return float(_psnr(_to_np(target), _to_np(pred), data_range=1.0))


def calculate_ssim(pred, target):
    return float(_ssim(_to_np(target), _to_np(pred),
                        multichannel=True, data_range=1.0, channel_axis=2))


def calculate_uqi(pred, target):
    p  = pred.cpu().numpy()[0].transpose(1, 2, 0)
    t  = target.cpu().numpy()[0].transpose(1, 2, 0)
    pm, tm = p.mean(), t.mean()
    cov    = np.mean((p - pm) * (t - tm))
    num    = 4 * cov * pm * tm
    den    = (np.var(p) + np.var(t)) * (pm ** 2 + tm ** 2)
    return float(num / den) if den != 0 else 0.0


def _rgb2y(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _patches(img, ws, stride):
    H, W = img.shape
    oh   = (H - ws) // stride + 1
    ow   = (W - ws) // stride + 1
    s    = (img.strides[0] * stride, img.strides[1] * stride, img.strides[0], img.strides[1])
    return np.lib.stride_tricks.as_strided(img, (oh, ow, ws, ws), s)


def _edge_weights(tp, eps=1e-10):
    gx  = np.abs(tp[:, :, :, 1:] - tp[:, :, :, :-1])
    gy  = np.abs(tp[:, :, 1:, :] - tp[:, :, :-1, :])
    return tp.var(axis=(-2, -1)) + gx.mean(axis=(-2, -1)) + gy.mean(axis=(-2, -1)) + eps


def calculate_oss_psnr(pred, target, ws=8, stride=4):
    p, t  = (_to_np(pred).astype(np.float32),
              _to_np(target).astype(np.float32))
    pp, tp = _patches(_rgb2y(p), ws, stride), _patches(_rgb2y(t), ws, stride)
    mse    = ((pp - tp) ** 2).mean(axis=(-2, -1))
    vals   = np.where(mse < 1e-10, 100.0, 10 * np.log10(1.0 / (mse + 1e-10)))
    wv     = _edge_weights(tp)
    return float((vals * (wv / wv.sum())).sum())


def calculate_oss_ssim(pred, target, ws=11, stride=4):
    p, t  = (_to_np(pred).astype(np.float32),
              _to_np(target).astype(np.float32))
    pp, tp = _patches(_rgb2y(p), ws, stride), _patches(_rgb2y(t), ws, stride)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mp, mt = pp.mean((-2, -1)), tp.mean((-2, -1))
    sp, st = pp.std((-2, -1)), tp.std((-2, -1))
    spt    = ((pp - mp[..., None, None]) * (tp - mt[..., None, None])).mean((-2, -1))
    vals   = ((2 * mp * mt + C1) * (2 * spt + C2)) / \
             ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))
    wv     = _edge_weights(tp)
    return float((vals * (wv / wv.sum())).sum())


def evaluate(model, loader, device, compute_oss=False):
    model.eval()
    psnrs, ssims, uqis, opsnrs, ossims = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            low  = batch['low'].to(device)
            high = batch['high'].to(device)
            pred, *_ = model(low)
            psnrs.append(calculate_psnr(pred, high))
            ssims.append(calculate_ssim(pred, high))
            uqis.append(calculate_uqi(pred, high))
            if compute_oss:
                opsnrs.append(calculate_oss_psnr(pred, high))
                ossims.append(calculate_oss_ssim(pred, high))
    out = {'psnr': np.mean(psnrs), 'ssim': np.mean(ssims), 'uqi': np.mean(uqis)}
    if compute_oss:
        out['oss_psnr'] = np.mean(opsnrs)
        out['oss_ssim'] = np.mean(ossims)
    return out
