import random

import torch

from .network import DQN


def test_dqn():
    arch = [(100, 100), (100, 100)]
    input_size = 28
    output_size = 2
    net = DQN(input_size, output_size, arch)
    assert len(net.linears) == len(arch)

    net.train()
    inp = torch.tensor([random.randint(0, 1) for _ in range(input_size)]).float()
    out = net(inp)
    assert out.size()[0] == output_size
