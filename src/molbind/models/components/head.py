from torch import nn
from typing import Union, List


class ProjectionHead(nn.Module):
    def __init__(self, dims, activation : Union[str, List[str]] = "leakyrelu"):
        super(ProjectionHead, self).__init__()
        # build projection head
        self.projection_head = self.build_projection_head(dims, activation)


    def build_projection_head(self, dims, activation):
        # Build projection head dynamically based on the length of dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))  # Optional: add batch normalization
            layers.append(self._get_activation(activation))  # Apply activation function

        return nn.Sequential(*layers)

    def _get_activation(self, activation):
            if isinstance(activation, str):
                if activation.lower() == 'relu':
                    return nn.ReLU(inplace=True)
                elif activation.lower() == 'leakyrelu':
                    return nn.LeakyReLU(inplace=True)
                elif activation.lower() == 'sigmoid':
                    return nn.Sigmoid()
                elif activation.lower() == 'tanh':
                    return nn.Tanh()
                else:
                    raise NotImplementedError(f"Activation {activation} is not implemented.")
            elif isinstance(activation, list):
                # TODO: is this compatible with the call above?
                # why more than one nonlinearity after the other?

                # In case you want multiple activation functions in sequence
                return nn.Sequential(*[self._get_activation(act) for act in activation])
            else:
                raise ValueError("Activation should be either a string or a list of strings.")