import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PairedLowLightDataset(Dataset):
    def __init__(self, root_dir, mode='train', scale_factor='X4'):
        self.root = Path(root_dir)
        self.mode = mode
        self.sf = scale_factor
        subset = 'Train' if mode == 'train' else 'Test'
        self.hr_dir = self.root / subset / 'HR'
        self.lr_dir = self.root / subset / 'LR_low' / scale_factor

        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        hr_imgs, lr_imgs = [], []
        for e in exts:
            hr_imgs.extend(sorted(self.hr_dir.glob(e)))
            lr_imgs.extend(sorted(self.lr_dir.glob(e)))

        self.pairs = self._match(hr_imgs, lr_imgs)
        self.aug = transforms.RandomHorizontalFlip(0.5) if mode == 'train' else None

    def _match(self, hr_imgs, lr_imgs):
        hr_dict = {p.stem: p for p in hr_imgs}
        pairs = []
        sfx = ['_x4', '_x8', '_X4', '_X8', 'x4', 'x8', '_lr', '_LR']
        for lp in lr_imgs:
            stem = lp.stem
            for s in sfx:
                if stem.endswith(s):
                    stem = stem[:-len(s)]
                    break
            if lp.stem in hr_dict:
                pairs.append((lp, hr_dict[lp.stem]))
            elif stem in hr_dict:
                pairs.append((lp, hr_dict[stem]))
        if not pairs:
            n = min(len(lr_imgs), len(hr_imgs))
            pairs = list(zip(lr_imgs[:n], hr_imgs[:n]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        lr = Image.open(lp).convert('RGB')
        hr = Image.open(hp).convert('RGB')
        if self.aug:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            lr = self.aug(lr)
            torch.manual_seed(seed)
            hr = self.aug(hr)
        to_t = transforms.ToTensor()
        return {'low': to_t(lr), 'high': to_t(hr), 'lr_path': str(lp), 'hr_path': str(hp)}
