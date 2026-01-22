# tensor_dataset.py
import torch
from torch.utils.data import Dataset

class CachedTensorDataset(Dataset):
    def __init__(self, tensor_path):
        self.X_num, self.X_cat, self.y = torch.load(tensor_path)

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
