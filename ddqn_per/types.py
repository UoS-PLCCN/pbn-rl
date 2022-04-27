from typing import Tuple, List

import torch


PERMinibatch = Tuple[
    # States, Actions, Rewards, Next States, Dones
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    # Indices, Weights
    List[int],
    torch.FloatTensor,
]

Minibatch = Tuple[
    # States, Actions, Rewards, Next States, Dones
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
]
