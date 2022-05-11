"""
memory.py - Module to hold Experience Replay-related functionality.

Heavily based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np

from .data_structures import MinSegmentTree, SumSegmentTree

#: Named tuple to model a transition sampled from the environment.
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ExperienceReplay:
    """An Experience Replay buffer."""

    def __init__(self, capacity: int):
        """An Experience Replay buffer.

        Args:
            capacity (int): the overall capacity of the buffer
        """
        # Data Store
        self.capacity = capacity
        self.buffer = []
        self.current_index = 0

    def clear(self):
        self.buffer = []
        self.current_index = 0

    def store(self, transition: Transition):
        """Store a new experience into the memory.

        Args:
            transition (Transition): the Transition to save.
        """
        # Add to main buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.current_index] = transition
        self.current_index = (self.current_index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample the Experience Replay Buffer randomly.

        Args:
            batch_size (int): the size of the batch to sample.

        Returns:
            List[Transition]: a random selection of transitions from the memory.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get the length of the Replay Buffer.

        Returns:
            int: the size of the replay buffer.
        """
        return len(self.buffer)


class PrioritisedER(ExperienceReplay):
    """A Prioritised Experience Replay buffer."""

    def __init__(self, capacity: int, prob_alpha: float = 0.6):
        """A Prioritised Experience Replay buffer.

        Args:
            capacity (int): the overall capacity of the buffer
            prob_alpha (float, optional): The prioritisation exponent alpha. Defaults to 0.6.
        """
        super().__init__(capacity)

        # Parameters
        self.alpha = prob_alpha

        # Data Store
        helper_buffer_size = 1
        while (
            helper_buffer_size < capacity
        ):  # Get power of 2 closest to (but higher than) the buffer size.
            helper_buffer_size *= 2
        self.sum_helper_buffer = SumSegmentTree(helper_buffer_size)
        self.weight_helper_buffer = MinSegmentTree(helper_buffer_size)

        # State variables
        self.max_priority = 1.0

    def clear(self):
        super().clear()
        self.sum_helper_buffer.clear()
        self.weight_helper_buffer.clear()
        self.max_priority = 1.0

    def store(self, transition: Transition):
        """Store a new experience into the memory.

        Args:
            transition (Transition): the Transition to save.
        """
        super().store(transition)
        # index = self.current_index - 1 if self.current_index > 0 else self.capacity - 1
        index = self.current_index

        # Store to helper buffers
        priority = self.max_priority**self.alpha
        self.sum_helper_buffer[index] = priority
        self.weight_helper_buffer[index] = priority

    def _sample_proportional(self, batch_size: int):
        batch = []
        p_total = self.sum_helper_buffer.sum(0, len(self.buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            index = self.sum_helper_buffer.find_prefixsum_index(mass)
            batch.append(index)
        return batch

    def sample(
        self, batch_size: int, beta=0.4
    ) -> Tuple[List[Transition], List[int], List[float]]:
        """Sample the Experience Replay Buffer proportionally.

        Args:
            batch_size (int): the size of the batch to sample.
            beta (float, optional): The bias correction exponent.
                Should be linearly annealed throughout training. Defaults to 0.4.

        Returns:
            tuple: (samples, indices, weights)
            samples (List[Transition]): the transitions sampled from the memory.
            indices (List[int]): the indices of the transitions sampled.
            weights (List[float]): the weights associated with the transitions
        """
        assert beta > 0
        N = len(self.buffer)

        # Probability calculation
        priority_sum = self.sum_helper_buffer.sum()
        probabilities = np.array(self.sum_helper_buffer.get_values(N)) / priority_sum

        # Proprotionally sample the buffer based on the probabilities
        indices = self._sample_proportional(batch_size)
        samples = [self.buffer[index] for index in indices]

        # Weight calculation
        min_probability = self.weight_helper_buffer.min() / priority_sum
        max_weight = (N * min_probability) ** (-beta)
        weights = ((N * probabilities[indices]) ** (-beta)) / max_weight
        weights = np.array(
            weights, dtype=np.float32
        )  # Convert to float32 from float64, apparently.

        return samples, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update the priorities of given experiences.

        Args:
            indices (List[int]): the indices of the experiences to update priorities for.
            priorities (List[float]): the new priorities for the experiences.
        """
        assert len(indices) == len(priorities)

        for i, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= i < len(self.buffer)

            priority_a = priority**self.alpha

            self.sum_helper_buffer[i] = priority_a
            self.weight_helper_buffer[i] = priority_a

            self.max_priority = max(self.max_priority, priority)
