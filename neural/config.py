from __future__ import annotations
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split

from dataclasses import dataclass, field
from utils.config_base import ConfigBase


@dataclass
class DatasetConfig(ConfigBase):
    input_frames: list[int] # The frames to use as input for the network. The frames are in the format of the number of frames before (negative) or after (positive) the prediction frame(0).
    pred_frames: list[int] # The frames to predict. The frames are in the format of the number of frames before (negative) or after (positive) the prediction frame(0).
    log_path: str # The path to the log file containing the worm head predictions (by YOLO).

    def __post_init__(self) -> None:
        if self.input_frames[0] != 0:
            print(
                "WARNING::DatasetConfig::frames_for_pred should contain 0 as first element. Please check verify you parameters."
            )

    @staticmethod
    def from_ioConfig(io: IOConfig, log_path: str) -> DatasetConfig:
        return DatasetConfig(io.input_frames, io.pred_frames, log_path)


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adamw": torch.optim.AdamW,
}

LOSSES = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
}


@dataclass
class TrainConfig(ConfigBase):
    # general parameters
    seed: int = field(default=42, kw_only=True)  # Random seed for reproducibility
    dataset: DatasetConfig  # The dataset to use for training, can also be a config object (if Dataset, it will be used as is)

    # trainer parameters
    model: nn.Module | str  # The model to train, can also be a pretrained model (if str, it will be loaded from disk)
    loss_fn: nn.Module  # The loss function to use, can be any of the keys in the LOSSES dict
    optimizer: str  # The optimizer to use, can be any of the keys in the OPTIMIZERS dict
    device: str = "cuda"  # 'cuda' for training on GPU or 'cpu' otherwise
    log: bool = False  # Whether to log and save the training process with tensorboard

    # training parameters
    num_epochs: int = 100  # Number of times to iterate over the dataset
    checkpoints: str = None  # Path to save model checkpoints, influding the checkpoint name.
    early_stopping: int = None  # Number of epochs to wait before stopping training if no improvement was made
    print_every: int = 5  # How often (#epochs) to print training progress

    # optimizer parameters
    learning_rate: float = 0.001  # Learning rate for the optimizer
    weight_decay: float = (
        1e-5  # Weight decay for the optimizer (regularization, values typically in range [0.0, 1e-4] but can be bigger)
    )

    # dataloader parameters
    batch_size: int = 256  # Number of samples in each batch
    shuffle: bool = True  # Whether to shuffle the dataset at the beginning of each epoch
    num_workers: int = 0  # Number of subprocesses to use for data loading
    train_test_split: float = 0.8  # Fraction of the dataset to use for training, the rest will be used for testing

    dl_train: DataLoader = field(init=False)
    dl_test: DataLoader = field(init=False)


@dataclass
class IOConfig(ConfigBase):
    """
    Configuration for the basic input/output of the network
    The input_frames and pred_frames are lists of integers that represent the frames
    that will be used as input and output of the network. The frames are in the format
    of the number of frames before (negative) or after (positive) the prediction frame(0).
    To calculate in_dim,out_dim we assume that each input frame has 4 features (x,y,w,h), representing the worm bbox in that frame
    and each prediction frame has 2 features (x,y), representing the worm center in that frame.
    """

    input_frames: list[int]
    pred_frames: list[int]

    in_dim: int = field(init=False)
    out_dim: int = field(init=False)

    def __post_init__(self):
        if 0 not in self.input_frames:
            print(
                "WARNING::IOConfig::__post_init__::input_frames doesn't contain 0 (the prediction frame). Please verify your parameters."
            )
        self.in_dim = len(self.input_frames) * 4
        self.out_dim = len(self.pred_frames) * 2

    @staticmethod
    def from_datasetConfig(config: DatasetConfig) -> IOConfig:
        return IOConfig(config.input_frames, config.pred_frames)


@dataclass
class ModuleConfig(ConfigBase):
    name: str
    args: dict[str, object]

    def initialize():
        pass


@dataclass
class NetworkConfig(ConfigBase):
    io: IOConfig
    architecture: list[ModuleConfig]
