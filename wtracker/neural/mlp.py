from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

from wtracker.neural.config import IOConfig

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


class WormPredictor(nn.Module):
    """
    A class that represents neural network models that predict worm behavior. After a model is created from several layers or blocks, it is wrapped in this class
    so that it can be distinguished from other models that don't predict worm behavior (for example the layers/blocks that make this model).
    This class also holds the IOConfig object that is used to determine the input and output shapes of the model, and the specific frames it expects as input and output.

    Attributes:
        model: The neural network model that predicts worm behavior.
        io_config: The IOConfig object of the model.
    """

    def __init__(self, model: nn.Module, io_config: IOConfig):
        super().__init__()
        self.io_config: IOConfig = io_config
        self.model: nn.Module = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MLPLayer(nn.Module):
    """
    A single layer perceptron, that can hold a bach-norm and activation layers as well.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Sequence[int],
        nonlin: Union[str, nn.Module],
        batch_norm: bool = True,
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
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.

        """
        return self.mlp_layer.forward(x.reshape(x.size(0), -1))


class MlpBlock(nn.Module):
    """
    A general-purpose MLP.

    Args:
        in_dim: Input dimension.
        dims: Hidden dimensions, including output dimension.
        nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
    """

    def __init__(
        self,
        in_dim: int,
        dims: Sequence[int],
        nonlins: Sequence[Union[str, nn.Module]],
        batch_norm: bool = True,
    ):
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        self.dims = dims
        self.nonlins = nonlins

        super().__init__()

        layers = []
        for i, out_dim in enumerate(self.dims):
            layers.append(MLPLayer(in_dim, out_dim, nonlins[i], batch_norm))
            in_dim = out_dim

        self.sequence = nn.Sequential(*layers)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return ACTIVATIONS[act](**ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.
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
        in_dim: int = None,  # if in_dim is an int, then a first layer will be made
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
            layers.append(MlpBlock(block_in_dim, block_dims, block_nonlins, batch_norm))

        self.blocks = nn.ModuleList(layers)
        # Create output layer
        self.output = nn.Linear(block_dims[-1], out_dim)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return ACTIVATIONS[act](**ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        x = self.input(x)
        for block in self.blocks:
            out = block(x)
            x = x + out
        return self.output(x)
