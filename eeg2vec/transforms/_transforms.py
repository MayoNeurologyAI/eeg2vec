import numpy as np
import pandas as pd
from collections import OrderedDict

class RandomTemporalCrop(object):

    def __init__(self, max_crop_frac=0.05, temporal_axis=1):
        """
        Uniformly crops the time-dimensions of a batch.

        Parameters
        ----------
        max_crop_frac: float
                       The is the maximum fraction to crop off of the trial.
        """
        assert 0 < max_crop_frac < 1
        self.max_crop_frac = max_crop_frac
        self.temporal_axis = temporal_axis

    def __call__(self, x):
        
        trial_len = x.shape[self.temporal_axis]
        crop_len = np.random.randint(int((1 - self.max_crop_frac) * trial_len), trial_len)
        offset = np.random.randint(0, trial_len - crop_len)
        
        x = x[:, offset:offset + crop_len, ...]

        return x
    
class UidToEpoch(object):
    
    def __init__(self, max_cache_size=100):
        
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
    
    @staticmethod  
    def _load_epoch_from_gcs(epoch_path):
    
        df = pd.read_pickle(epoch_path)
        
        return np.stack(df['data']).squeeze()
        
    def __call__(self, sample):
        
        uid = sample['uid']
        path = sample['path']
        
        if uid not in self.cache:
            self.cache[path] = UidToEpoch._load_epoch_from_gcs(path)
            
        # If cache size exceeds the max limit, remove the oldest item
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # pop the first item
            
        epoch = self.cache[path]
        
        return epoch
