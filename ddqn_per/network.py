"""
network.py - This module holds the Agent's DQN.
"""
from torch import nn
from torch.nn import functional as F

from typing import List
from numbers import Number


class DQN(nn.Module):
    """Neural Network used as the DQN."""

    def __init__(self, input_size: int, output_size: int, net_arch: list):
        """Neural network to approximate Q values.

        Args:
            input size (int): number of nodes of the input layer. The size of the PBN.
            output_size (int): number of nodes in the output layer. The number of actions. Size + 1
            height (int): number of nodes in the hidden layer of the NN. Arbitrary.
        """
        super().__init__()
        self.input = nn.Linear(input_size, net_arch[0][0], bias=True)
        self.linears = nn.ModuleList(
            [nn.Linear(arch[0], arch[1], bias=True) for arch in net_arch]
        )
        self.output = nn.Linear(self.linears[-1].out_features, output_size, bias=True)

    def forward(self, x: List[Number]) -> List[float]:
        """A forward-pass of the neural network.

        Args:
            x (List[Number]): Network input. The PBN state in this case.

        Returns:
            List[float]: The network output. Value at index A is the expected cumulative reward if action A is taken.
        """
        x = F.relu(self.input(x))
        for linear in self.linears:
            x = F.relu(linear(x))
        return self.output(x)
