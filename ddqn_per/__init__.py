"""
main.py - This module holds the actual Agent.
"""
import random

import torch
from torch import optim
from torch.nn import functional as F

from gym_PBN.envs.pbn_target import PBNTargetEnv

from .memory import ExperienceReplay, PrioritisedER, Transition
from .network import DQN
from .types import Minibatch, PERMinibatch


class DDQN:
    """The agent of the RL algorithm. Houses the DQN, ER, etc."""

    def __init__(
        self,
        env: PBNTargetEnv = None,
        device: torch.device = "cpu",
        policy_kwargs: dict = {"net_arch": [(100, 100), (100, 100)]},
        gamma=0.01,
    ):
        self.device = device

        # The size of the PBN
        # HACK only works for our current env
        self.input_size = env.observation_space.n
        self.output_size = env.action_space.n

        self.env = env

        # Networks
        self.policy_kwargs = policy_kwargs
        self.controller = DQN(self.input_size, self.output_size, **policy_kwargs).to(
            self.device
        )
        self.target = DQN(self.input_size, self.output_size, **policy_kwargs).to(
            self.device
        )
        self.target.load_state_dict(self.controller.state_dict())

        # Reinforcement learning parameters
        self.gamma = gamma

        # State
        self.train = False

    @classmethod
    def load(cls, path, env: PBNTargetEnv = None, device: torch.device = "cpu"):
        state_dict = torch.load(path)
        agent = cls(
            env,
            device,
            gamma=state_dict["gamma"],
            policy_kwargs=state_dict["policy_kwargs"],
        )
        agent.controller.load_state_dict(state_dict["model"])
        agent.target.load_state_dict(state_dict["model"])

    def save(self, path):
        state_dict = {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "gamma": self.gamma,
            "epsilon": self.EPSILON,
            "steps": self.train_steps,
        }

        with open(path) as f:
            torch.save(state_dict, f)

    def _get_learned_action(self, state) -> int:
        with torch.no_grad():
            q_vals = self.controller(
                torch.tensor(state, device=self.device, dtype=torch.float)
            )
            action = q_vals.max(0)[1].view(1, 1).item()
        return action

    def predict(self, state, deterministic: bool = True) -> int:
        if self.train and random.uniform(0, 1) <= self.EPSILON:
            return random.choice(self.actions)
        else:
            return self._get_learned_action(state)

    def _fetch_minibatch(self) -> Minibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            Minibatch: a minibatch.
        """
        # Fetch data
        experiences = self.replay_memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, dones = zip(
            *experiences
        )

        # Load to device
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)

        return (state_batch, action_batch, reward_batch, next_state_batch, dones)

    def _get_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Get huber loss based on a batch of experiences.

        Args:
            states (torch.Tensor): the batch of states.
            actions (torch.Tensor): the batch of agent.
            rewards (torch.Tensor): the batch of rewards received.
            next_states (torch.Tensor): the batch of the resulting states.
            dones (torch.Tensor): the batch of done flags.
            reduction (str, optional): the reduction to use on the loss.

        Returns:
            torch.Tensor: the huber loss as a tensor.
        """
        # Calculate predicted actions
        with torch.no_grad():
            vals = self.controller(next_states)  # TODO Wait shouldn't this be target?
            action_prime = vals.max(1)[1].unsqueeze(1)

        # Calculate current and target Q to calculate loss
        controller_Q = self.controller(states).gather(1, actions)
        target_Q = rewards + (1 - dones) * self.gamma * self.target(next_states).gather(
            1, action_prime
        )
        return F.smooth_l1_loss(controller_Q, target_Q, reduction=reduction)

    def _back_propagate(self, loss: torch.Tensor):
        """Do a step of back propagation based on a loss vector.

        Args:
            loss (torch.Tensor): the loss vector as a tensor.
        """
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def learn(self, steps):
        # TODO params
        self.toggle_train(steps, ...)

        # TODO init environment
        for step in steps:
            # TODO training loop
            ...

        if len(self.replay_memory) >= self.batch_size:
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                dones,
            ) = self._fetch_minibatch()
            loss = self._get_loss(
                state_batch, action_batch, reward_batch, next_state_batch, dones
            )
            self._back_propagate(loss)
        self.train_count += 1

        # Oh yeah after every TARGET_UPDATE steps the target network gets updated.
        if self.train_count % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.controller.state_dict())

    def feedback(self, transition: Transition, learn: bool = True):
        """Save an experience tuple, do a step of back propagation.

        Args:
            transition (Transition): Transition :)
            learn (bool, optional): Whether or not to do a step of back propagation.
        """
        self.replay_memory.store(transition)
        if learn:
            self.learn()

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    def toggle_train(
        self,
        steps: int,
        batch_size: int,
        memory_size: int,
        min_epsilon: float,
        target_update: int,
    ):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        self.train = True
        self.train_steps = steps
        self.train_count = 0

        self.optimizer = optim.RMSprop(self.controller.parameters())

        # Memory
        self.batch_size = batch_size
        self.replay_memory = ExperienceReplay(memory_size)

        # Explore-exploit
        self.EPSILON = 1
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = min_epsilon
        self.EPSILON_DECREMENT = (
            self.MAX_EPSILON - self.MIN_EPSILON
        ) / self.train_steps

        self.TARGET_UPDATE = target_update


class DDQNPER(DDQN):
    """Agent using Prioritized Experience Replay."""

    def _fetch_minibatch(self) -> PERMinibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            PERMinibatch: a minibatch.
        """
        # Fetch data
        experiences, indices, weights = self.replay_memory.sample(
            self.batch_size, self.BETA
        )
        state_batch, action_batch, reward_batch, next_state_batch, dones = zip(
            *experiences
        )

        # Load to device
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)
        weights = (
            torch.tensor(weights, device=self.device, dtype=torch.float)
            .squeeze()
            .unsqueeze(1)
        )

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            dones,
            indices,
            weights,
        )

    def learn(self):
        """Sample a minibatch of experiences, do a step of back propagation."""
        if len(self.replay_memory) >= self.batch_size:
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                dones,
                indices,
                weights,
            ) = self._fetch_minibatch()
            loss = self._get_loss(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                dones,
                reduction="none",
            )
            loss *= weights

            # Update priorities in the PER buffer
            priorities = loss + self.REPLAY_CONSTANT
            # TODO Wait don't these have to be positive? Maybe need to abs()?
            self.replay_memory.update_priorities(
                indices, priorities.data.detach().squeeze().cpu().numpy().tolist()
            )

            # Back propagation
            loss = loss.mean()
            self._back_propagate(loss)
        self.train_count += 1

        # Oh yeah after every TARGET_UPDATE steps the target network gets updated.
        if self.train_count % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.controller.state_dict())

    def increment_beta(self):
        """Increment the beta exponent."""
        self.BETA = min(self.BETA + self.BETA_INCREMENT_CONSTANT, 1)

    def toggle_train(self, **kwargs):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        super().toggle_train(**kwargs)

        # PER
        self.REPLAY_CONSTANT = 1e-5
        self.BETA = 0.4
        self.BETA_INCREMENT_CONSTANT = self.BETA / (0.75 * self.train_steps)
        self.replay_memory = PrioritisedER(kwargs["memory_size"])
