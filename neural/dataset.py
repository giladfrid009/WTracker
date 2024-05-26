from __future__ import annotations
from torch.utils.data import Dataset
from torch import Tensor
import torch
import pandas as pd
import numpy as np

from neural.config import DatasetConfig
from utils.bbox_utils import BoxUtils


class numpyDataset(Dataset):
    """
    A custom Dataset class used to train the neural network. This class is used to create a PyTorch Dataset from a numpy array, and
    can be initialized with 'ndarrays' of the samples and labels, as well as a DatasetConfig configuration, in which the samples (X)
    and labels(y) will be created automatically.

    Args:
        X (np.ndarray): The input data as a numpy array.
        y (np.ndarray): The output data as a numpy array.
        config (DatasetConfig, optional): The configuration object for the dataset. Defaults to None.

    Attributes:
        config (DatasetConfig): The configuration object for the dataset.
        X (Tensor): The input data as a PyTorch tensor.
        y (Tensor): The output data as a PyTorch tensor.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the input and output data at the given index.
        save(path): Saves the dataset to a file.
        load(path): Loads a saved dataset from a file.
        create_from_config(config, save_path): Creates a new dataset from a DatasetConfig configuration object.

    """

    def __init__(self, X: np.ndarray, y: np.ndarray, config: DatasetConfig = None):
        self.config = config
        self.X = Tensor(X)
        self.y = Tensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]

    def save(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> None:
        return torch.load(path)

    @staticmethod
    def create_from_config(config: DatasetConfig, save_path: str | None = None) -> numpyDataset:
        data = pd.read_csv(config.log_path)
        start_idx = abs(min(config.input_frames)) + 1
        X_mask = np.asanyarray(config.input_frames)
        y_mask = np.asanyarray(config.pred_frames)

        wrm_boxes = data[["wrm_x", "wrm_y", "wrm_w", "wrm_h"]].to_numpy(dtype=np.float64)
        wrm_centers = BoxUtils.center(wrm_boxes)

        # Create columns for X and y
        X_cols_prefix = ["wrm_x", "wrm_y", "wrm_w", "wrm_h"]
        y_cols_prefix = ["wrm_center_x", "wrm_center_y"]
        X_cols = []
        y_cols = []
        for i in config.input_frames:
            X_cols += [col + str(i) for col in X_cols_prefix]
        for i in config.pred_frames:
            y_cols += [col + str(i) for col in y_cols_prefix]

        # Create X and y
        X = pd.DataFrame(index=data.index, columns=X_cols)
        y = pd.DataFrame(index=data.index, columns=y_cols)
        for i in range(start_idx, len(data) - max(config.pred_frames) - 1):
            X.iloc[i] = wrm_boxes[i + X_mask].reshape(1, -1)
            y.iloc[i] = wrm_centers[i + y_mask].reshape(1, -1)

        # Drop rows with NaN values
        na_mask = np.ma.mask_or(X.isna().any(axis=1), y.isna().any(axis=1))
        X = X.loc[~na_mask]
        y = y.loc[~na_mask]

        X = X.to_numpy(dtype=np.float32, copy=True)
        y = y.to_numpy(dtype=np.float32, copy=True)

        # make X and y coordinates relative to the prediction frame
        x_cords = X[:, 0].reshape(-1, 1)
        y_cords = X[:, 1].reshape(-1, 1)

        x_cord_mask = np.arange(y.shape[1]) % 2 == 0
        y_cord_mask = np.arange(y.shape[1]) % 2 == 1
        y[:, x_cord_mask] -= x_cords
        y[:, y_cord_mask] -= y_cords

        x_cord_mask = np.arange(X.shape[1]) % 4 == 0
        y_cord_mask = np.arange(X.shape[1]) % 4 == 1
        X[:, x_cord_mask] -= x_cords  #
        X[:, y_cord_mask] -= y_cords  # .reshape(-1, 1)

        dataset = numpyDataset(X, y, config)
        
        if save_path is not None:
            dataset.save(save_path)

        return dataset
