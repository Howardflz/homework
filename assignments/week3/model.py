import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    """
    Structure of the network.

    Arguments:
        __init__: initialize the parameters.
        forward: the forward process of NN

    Returns:
        Nothing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = nn.ReLU,
        initializer: Callable = nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation()
        self.initializer = initializer

        for i in range(hidden_count):
            next_num_inputs = hidden_size
            self.layer = nn.Linear(input_size, next_num_inputs)
            self.initializer(self.layer.weight)
            self.layers += [self.layer, self.activation]
            input_size = next_num_inputs

        self.out = nn.Linear(input_size, num_classes)
        # self.batchNormal = nn.BatchNorm1d(self.hidden_size)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        for layer in self.layers:
            x = layer(x)
            # x = self.batchNormal(x)
            # x = self.dropout(x)

        x = self.out(x)
        return x
