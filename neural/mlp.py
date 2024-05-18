import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        self.dims = dims
        self.nonlins = nonlins

        super().__init__()

        layers = []
        for i, out_dim in enumerate(self.dims):
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(self._make_activation(nonlins[i]))
        
        layers[-2] = nn.Identity()
        layers[-1] = nn.Identity()

        self.sequence = nn.Sequential(*layers)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return ACTIVATIONS[act](**ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        return self.sequence.forward(x.reshape(x.size(0), -1))
