import os
import time
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from fuzzy_restoration import (
    ConvNeXtFuzzyModel, PairedLowLightDataset, ICMoELoss,
    evaluate, save_checkpoint, load_checkpoint,
)

torch.backends.cudnn.benchmark = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--scale', default='X4', choices=['X4', 'X8'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--ckpt_dir', default='./checkpoints')
    p.add_argument('--resume', default=None)
    return p.parse_args()

def build_config(args):
    return {
        'scale_factor': args.scale,
        'feature_channels': 64,
        'n_clusters': 5,
        'n_experts': 4,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'warmup_epochs': 5,
        'eval_every': 5,
        'lambda_recon': 1.0,
        'lambda_percep': 0.05,
        'lambda_adv': 0.005,
        'lambda_illum': 0.1,
        'lambda_util': 0.05,
        'lambda_ent': 0.01,
    }

def main():
    args = parse_args()
    config = build_config(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    full_train = PairedLowLightDataset(args.data_root, mode='train', scale_factor=args.scale)
    test_set = PairedLowLightDataset(args.data_root, mode='test', scale_factor=args.scale)

    n_val = max(1, int(len(full_train) * 0.10))
    n_tr = len(full_train) - n_val
    train_sub, val_sub = random_split(full_train, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    dl_kw = dict(num_workers=args.num_workers, pin_memory=True,
                 persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **dl_kw)

    model = ConvNeXtFuzzyModel(config).to(device)
    criterion = ICMoELoss(config).to(device)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    illum_ids = {id(p) for p in model.sa_anfille.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in illum_ids], 'lr': config['lr'] * 0.2},
        {'params': [p for p in model.parameters() if id(p) not in illum_ids], 'lr': config['lr']},
    ], betas=(0.9, 0.999), weight_decay=5e-5)

    we = config['warmup_epochs']
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, we),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'] - we, 1e-6),
    ], milestones=[we])

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, device, optimizer, scaler) + 1

    best_psnr = 0.0
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        ep_loss, n_batch, t0 = 0.0, 0, time.time()

        for batch in train_loader:
            low = batch['low'].to(device, non_blocking=True)
            high = batch['high'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                pred, illum, e_mem = model(low)
                losses = criterion(pred, high, illum, e_mem, discriminator=model.refinement.discriminator)

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += losses['total'].item()
            n_batch += 1

        scheduler.step()
        val_m = evaluate(model, val_loader, device, compute_oss=False)

        if val_m['psnr'] > best_psnr:
            best_psnr = val_m['psnr']
            best_path = os.path.join(args.ckpt_dir, f"best_fuzzy_{args.scale}.pth")
            save_checkpoint(best_path, epoch, model, optimizer, scaler, best_psnr, config)

        is_test = ((epoch + 1) % config['eval_every'] == 0 or epoch == config['num_epochs'] - 1 or epoch == 0)
        if is_test:
            test_m = evaluate(model, test_loader, device, compute_oss=True)

        if (epoch + 1) % 10 == 0:
            ep_path = os.path.join(args.ckpt_dir, f"epoch{epoch+1}_{args.scale}.pth")
            save_checkpoint(ep_path, epoch, model, optimizer, scaler, val_m['psnr'], config)

if __name__ == '__main__':
    main()
