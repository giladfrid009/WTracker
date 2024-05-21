from __future__ import annotations
from torch.utils.data import Dataset
from torch import Tensor
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.config_base import ConfigBase
from evaluation.analysis import Plotter
from config import DatasetConfig


class numpyDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray, config:DatasetConfig=None):
        self.config = config
        self.X = Tensor(X)
        self.y = Tensor(y)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]

    def save(self, path:str) -> None:
        torch.save(self, path)
    
    @staticmethod
    def load(path:str) -> None:
        return torch.load(path)

    @staticmethod
    def create_from_config(config:DatasetConfig, save_path:str=None) -> numpyDataset:
        data = pd.read_csv(config.log_path)
        start_idx = abs(min(config.input_frames)) + 1
        X_mask = np.asanyarray(config.input_frames)
        y_mask = np.asanyarray([config.pred_frame])

        data = Plotter.concat(data, Plotter.calc_centers(data, "wrm"))
        
        wrm_centers = data[['wrm_center_x', 'wrm_center_y']].to_numpy(dtype=np.float64)
        wrm_boxes = data[['wrm_x', 'wrm_y', 'wrm_w', 'wrm_h']].to_numpy(dtype=np.float64)
        
        # Create columns for X and y
        cols = ['wrm_x', 'wrm_y', 'wrm_w', 'wrm_h']
        y_cols = ['wrm_center_x', 'wrm_center_y']
        X_cols = []
        for i in config.input_frames:
            X_cols += [col+str(i) for col in cols]
        
        # Create X and y
        X = pd.DataFrame(index=data.index, columns=X_cols)
        y = pd.DataFrame(index=data.index, columns=y_cols)
        for i in range(start_idx, len(data)-config.pred_frame-1):
            X.iloc[i] = wrm_boxes[i + X_mask].reshape(1, -1)
            y.iloc[i] = wrm_centers[i + y_mask].reshape(1, -1)

        # Drop rows with NaN values
        na_mask = np.ma.mask_or(X.isna().any(axis=1), y.isna().any(axis=1))
        X = X.loc[~na_mask]
        y = y.loc[~na_mask]
        
        X = X.to_numpy(dtype=np.float32, copy=True)
        y = y.to_numpy(dtype=np.float32, copy=True)

        # make X and y coordinates relative to the prediction frame
        x_cord_mask = np.arange(X.shape[1]) % 4 == 0
        y_cord_mask = np.arange(X.shape[1]) % 4 == 1
        
        x_cords = X[:, 0].reshape(-1, 1)
        y_cords = X[:, 1].reshape(-1, 1)

        y[:, [0]] -= x_cords
        y[:, [1]] -= y_cords
        X[:, x_cord_mask] -= x_cords#
        X[:, y_cord_mask] -= y_cords#.reshape(-1, 1)
        
        dataset = numpyDataset(X, y, config)
        if save_path is not None:
            dataset.save(save_path)

        return dataset




