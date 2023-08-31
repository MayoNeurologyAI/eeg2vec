import torch
import numpy as np
from joblib import Parallel, delayed

class CollateEpochs(object):
    def __init__(self, transform, jobs=-1):
        self.transform = transform
        self.jobs = jobs
    
    def __call__(self, batch):
        transforms = Parallel(n_jobs=self.jobs)(delayed(self.transform)(sample) for sample in batch)
        return torch.from_numpy(np.stack(transforms))