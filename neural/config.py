import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from typing import Any, Tuple, Callable, Optional, cast, Union

from dataclasses import dataclass
from utils.config_base import ConfigBase


@dataclass
class DatasetConfig(ConfigBase):
    input_frames: list[int]
    pred_frame: int
    log_path: str

    def __post_init__(self):
        if self.input_frames[0] != 0:
            print("WARNING::DatasetConfig::frames_for_pred should contain 0 as first element. Please check verify you parameters.")


@dataclass
class TrainConfig(ConfigBase):
    # general parameters
    seed: int = 42 # Random seed for reproducibility
    dataset:Dataset

    # trainer parameters
    model: nn.Module # The model to train, can also be a pretrained model
    loss_fn: Union[nn.Module, str] # The loss function to use
    optimizer: Union[Optimizer, str] # The optimizer to use
    device: str = 'cuda' # 'cuda' for training on GPU or 'cpu' otherwise

    # training parameters
    num_epochs: int = 100 # Number of times to iterate over the dataset
    checkpoints: str = None # Path to save model checkpoints
    early_stopping: int = None, # Number of epochs to wait before stopping training if no improvement was made
    print_every: int = 1, # How often (#epochs) to print training progress
    
    # optimizer parameters
    learning_rate: float = 0.001 # Learning rate for the optimizer
    weight_decay: float = 0.0 # Weight decay for the optimizer (regularization, values typically in range [0.0, 1e-4] but can be bigger)
    
    # dataloader parameters
    batch_size: int = 64 # Number of samples in each batch
    shuffle: bool = True # Whether to shuffle the dataset at the beginning of each epoch
    num_workers: int = 0 # Number of subprocesses to use for data loading
    train_test_split: float = 0.8 # Fraction of the dataset to use for training, the rest will be used for testing
    
    

    

    def __post_init__(self):
        pass
        # self.device = torch.device(self.device)












