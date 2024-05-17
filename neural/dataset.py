from torch.utils.data import Dataset
from torch import Tensor
import pandas as pd
import numpy as np




class numpyDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray):
        # self.data = pd.read_csv(annotations_file, usecols=features_cols+label_cols)
        self.X = Tensor(X)
        self.y = Tensor(y)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]


class DataframeDataset(Dataset):
    def __init__(self, df:pd.DataFrame, features_cols:list[str], label_cols:list[str], transform=None, target_transform=None):
        # self.data = pd.read_csv(annotations_file, usecols=features_cols+label_cols)
        self.X = Tensor(df.to_numpy(dtype=np.float32))
        self.y = Tensor(df.to_numpy(dtype=np.float32))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]
