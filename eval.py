import os
import sys
import glob
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzy_restoration import ConvNeXtFuzzyModel, PairedLowLightDataset, evaluate
from fuzzy_restoration.utils.checkpoint import load_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--scale', default='X4', choices=['X4', 'X8'])
    p.add_argument('--all_dlevels', action='store_true')
    p.add_argument('--img_size', type=int, default=256)
    return p.parse_args()

def build_model(scale, device):
    config = {'scale_factor': scale, 'feature_channels': 64, 'n_clusters': 5, 'n_experts': 4}
    return ConvNeXtFuzzyModel(config).to(device)

def eval_one(model, data_dir, scale, device):
    try:
        ds = PairedLowLightDataset(data_dir, mode='test', scale_factor=scale)
    except (FileNotFoundError, RuntimeError) as e:
        return None
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return evaluate(model, loader, device, compute_oss=True)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = build_model(args.scale, device)
    load_checkpoint(args.ckpt, model, device)
    model.eval()

    if args.all_dlevels:
        roots = sorted(glob.glob(os.path.join(args.data_root, 'd[0-9]')))
        if not roots:
            roots = sorted(glob.glob(os.path.join(args.data_root, 'd[0-9][0-9]')))
    else:
        roots = [args.data_root]

    for root in roots:
        tag = os.path.relpath(root, args.data_root) if args.all_dlevels else os.path.basename(root)
        m = eval_one(model, root, args.scale, device)
        if m is None:
            continue
        print(f"{tag:<33} {m['psnr']:>8.4f} {m['ssim']:>8.4f} {m['uqi']:>8.4f} {m['oss_psnr']:>10.4f} {m['oss_ssim']:>10.4f}")

if __name__ == '__main__':
    main()
