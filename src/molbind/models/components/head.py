from torch import nn
from typing import Union, List
from class_resolver import ClassResolver

ACTIVATION_RESOLVER = ClassResolver(
    [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh], base=nn.Module, default=nn.ReLU
)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        dims,
        activation: Union[str, List[str]] = "leakyrelu",
        batch_norm: bool = False,
    ):
        super(ProjectionHead, self).__init__()
        # build projection head
        self.projection_head = self.build_projection_head(dims, activation, batch_norm)

    def build_projection_head(
        self, dims, activation, batch_norm=False
    ) -> nn.Sequential:
        # Build projection head dynamically based on the length of dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Optional: add batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            # Apply activation function
            layers.append(self._get_activation(activation))

        return nn.Sequential(*layers)

    def _get_activation(self, activation: Union[str, List[str]]):
        if isinstance(activation, str):
            return ACTIVATION_RESOLVER.resolve(activation)
        elif isinstance(activation, list):
            # In case you want multiple activation functions in sequence
            return nn.Sequential(*[self._get_activation(act) for act in activation])
        else:
            raise ValueError(
                "Activation should be either a string or a list of strings."
            )

    def forward(self, x):
        return self.projection_head(x)
