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

class MLPLayer(nn.Module):
    """
    A single layer perceptron, that can hold a bach-norm and activation layers as well.
    """

    def __init__(
        self, in_dim: int, out_dim: Sequence[int], nonlin: Union[str, nn.Module], batch_norm: bool = True
    ) -> None:
        super().__init__()

        layers = []

        layers.append(nn.Linear(in_dim, out_dim))
        in_dim = out_dim
        if batch_norm and nonlin not in ["none", None]:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(self._make_activation(nonlin))

        self.mlp_layer = nn.Sequential(*layers)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return ACTIVATIONS[act](**ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        return self.mlp_layer.forward(x.reshape(x.size(0), -1))

class MlpBlock(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self,
        in_dim: int,
        dims: Sequence[int],
        nonlins: Sequence[Union[str, nn.Module]],
        batch_norm: bool = True,
    ):
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
            layers.append(MLPLayer(in_dim, out_dim, nonlins[i], batch_norm))
            # layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            # if batch_norm and nonlins[i] not in ["none", None]:
                # layers.append(nn.BatchNorm1d(out_dim))
            # layers.append(self._make_activation(nonlins[i]))

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


class RMLP(nn.Module):
    def __init__(
        self,
        block_in_dim: int,
        block_dims: Sequence[int],
        block_nonlins: Sequence[Union[str, nn.Module]],
        n_blocks: int,
        out_dim: int,
        in_dim: int=None, # if in_dim is an int, then a first layer will be made
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        
        # Create first layer if in_dim is not None
        self.input = nn.Identity()
        if in_dim is not None:
            self.input = MLPLayer(in_dim, block_in_dim, block_nonlins[0], batch_norm)
        
        # Create blocks
        layers = []
        for i in range(n_blocks):
            layers.append(
                MlpBlock(block_in_dim, block_dims, block_nonlins, batch_norm)
            )

        self.blocks = nn.ModuleList(layers)
        # Create output layer
        self.output = nn.Linear(block_dims[-1], out_dim)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return ACTIVATIONS[act](**ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        x = self.input(x)
        for block in self.blocks:
            out = block(x)
            x = x + out
        return self.output(x)










