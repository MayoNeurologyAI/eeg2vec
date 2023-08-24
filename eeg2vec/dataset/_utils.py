import torch
from torch.utils.data import Dataset


class EpochDataset(Dataset):
    
    def __init__(self, data, transform):
        
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.df.index[idx]
        path = self.df.iloc[idx]['file_path']
        
        sample = {
            'uid' : uid,
            'path' : path
        }
        
        return self.transform(sample)




