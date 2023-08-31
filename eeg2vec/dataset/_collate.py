import torch
import numpy as np
from joblib import Parallel, delayed

class CollateEpochs(object):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, batch):
        transforms = Parallel(n_jobs=-1)(delayed(self.transform)(sample) for sample in batch)
        return torch.from_numpy(np.stack(transforms))