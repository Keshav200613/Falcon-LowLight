import os
import torch


def get_clean_state_dict(model):
    sd = model.state_dict()
    return {(k[10:] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}


def save_checkpoint(path, epoch, model, optimizer, scaler, val_psnr, config):
    torch.save({
        'epoch'              : epoch,
        'model_state_dict'   : get_clean_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict'  : scaler.state_dict() if scaler else None,
        'val_psnr'           : val_psnr,
        'config'             : config,
    }, path)


def load_checkpoint(path, model, device, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd   = ckpt.get('model_state_dict', ckpt)
    sd   = {(k[10:] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', -1)
