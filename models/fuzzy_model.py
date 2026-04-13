import torch.nn as nn
from .sa_anfille import DeepFeatureExtractor, SpatiallyAdaptiveANFILLE
from .restoration import ConvNeXtExpertRestoration
from .dict_sr import DictionaryBasedSR
from .discriminator import AdversarialRefinement

class ConvNeXtFuzzyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        fc = config['feature_channels']
        sf = int(str(config['scale_factor']).replace('X', '').replace('x', ''))

        self.dfe = DeepFeatureExtractor(3, fc)
        self.sa_anfille = SpatiallyAdaptiveANFILLE(fc, config['n_clusters'], config['n_experts'])
        self.restoration = ConvNeXtExpertRestoration(fc, n_rrg=3, n_experts=config['n_experts'])
        self.dict_sr = DictionaryBasedSR(scale_factor=sf, in_channels=fc, n_atoms=64, n_components=4)
        self.refinement = AdversarialRefinement()

    def forward(self, x_lr):
        F_base = self.dfe(x_lr)
        illum_map, memberships, e_mem = self.sa_anfille(F_base)
        restored = self.restoration(F_base, illum_map, e_mem)
        sr_output = self.dict_sr(restored, illum_map)
        final = self.refinement(sr_output)
        return final, illum_map, e_mem
