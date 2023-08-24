import numpy as np
import pandas as pd
from collections import OrderedDict

class RandomTemporalEndCrop:

    def __init__(self, end_crop_frac=0.25, crop_weights=None, temporal_axis=1):
        self.end_crop_frac = end_crop_frac
        self.crop_weights = np.array(crop_weights)
        self.temporal_axis = temporal_axis

    def __call__(self, x, training=False):
        if not training:
            return x
        if self.crop_weights is None:
            assert 0 <= self.end_crop_frac <= 1
            self.crop_weights = np.ones(int(x.shape[self.temporal_axis] * self.end_crop_frac))

        no_crop_len = x.shape[self.temporal_axis] - len(self.crop_weights)
        assert no_crop_len >= 0
        inds = np.arange(no_crop_len, x.shape[self.temporal_axis])
        crop_location = np.random.choice(inds, p=self.crop_weights / self.crop_weights.sum())
        return x[:, :crop_location, ...]
    
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
            self.cache[uid] = UidToEpoch._load_epoch_from_gcs(uid)
            
        # If cache size exceeds the max limit, remove the oldest item
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # pop the first item
            
        epoch = self.cache[uid]
        
        return {'epoch' : epoch, 'uid' : uid, 'path' : path}
