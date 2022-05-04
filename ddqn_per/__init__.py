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
        buffer_size: int = 5120,
        batch_size: int = 128,
        target_update: int = 1000,
        gamma=0.01,
        horizon=11,
    ):
        self.device = device

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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.EPSILON = 1
        self.TARGET_UPDATE = target_update
        self.horizon = horizon

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
            buffer_size=state_dict["buffer_size"],
            batch_size=state_dict["batch_size"],
            target_update=state_dict["target_update"],
        )
        agent.controller.load_state_dict(state_dict["model"])
        agent.target.load_state_dict(state_dict["model"])
        agent.EPSILON = state_dict["epsilon"]
        agent.current_episode = state_dict["current_episode"]

    def save(self, path):
        state_dict = {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "gamma": self.gamma,
            "epsilon": self.EPSILON,
            "current_episode": self.current_episode,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update": self.TARGET_UPDATE,
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

    def predict(self, state, deterministic: bool = False) -> int:
        if self.train and not deterministic and random.uniform(0, 1) <= self.EPSILON:
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
            vals = self.controller(next_states)
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

    def learn(
        self, total_episodes, checkpoint_freq: int = 25_000, checkpoint_path=None
    ):
        self.toggle_train(total_episodes)
        total_steps = 0

        for _ in range(self.current_episode, total_episodes):
            self.env.reset()
            current_step, done = 0, False

            while not done and current_step < self.horizon:
                total_steps += 1
                state = self.env.render()
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.store(
                    Transition(state, reward, action, next_state, done)
                )

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

                if total_steps % self.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.controller.state_dict())

            self.current_episode += 1
            if self.current_episode % checkpoint_freq == 0:
                self.save(checkpoint_path)

            self.decrement_epsilon()

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    def toggle_train(
        self,
        total_episodes: int,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1,
    ):
        """Setting all of the training params."""
        self.train = True
        self.train_episodes = total_episodes

        self.optimizer = optim.RMSprop(self.controller.parameters())

        # Memory
        self.replay_memory = ExperienceReplay(self.buffer_size)

        # Explore-exploit
        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.EPSILON_DECREMENT = (
            self.MAX_EPSILON - self.MIN_EPSILON
        ) / self.train_episodes


class DDQNPER(DDQN):
    """Agent using Prioritized Experience Replay."""

    def __init__(
        self,
        env: PBNTargetEnv = None,
        device: torch.device = "cpu",
        policy_kwargs: dict = {"net_arch": [(100, 100), (100, 100)]},
        buffer_size: int = 5120,
        batch_size: int = 128,
        target_update: int = 1000,
        gamma=0.01,
    ):
        super().__init__(
            env, device, policy_kwargs, buffer_size, batch_size, target_update, gamma
        )
        self.replay_memory = PrioritisedER(self.buffer_size)
        self.BETA = 0.4

    @classmethod
    def load(cls, path, env: PBNTargetEnv = None, device: torch.device = "cpu"):
        state_dict = torch.load(path)
        agent = cls(
            env,
            device,
            gamma=state_dict["gamma"],
            policy_kwargs=state_dict["policy_kwargs"],
            buffer_size=state_dict["buffer_size"],
            batch_size=state_dict["batch_size"],
            target_update=state_dict["target_update"],
        )
        agent.controller.load_state_dict(state_dict["model"])
        agent.target.load_state_dict(state_dict["model"])
        agent.EPSILON = state_dict["epsilon"]
        agent.BETA = state_dict["beta"]
        agent.current_episode = state_dict["current_episode"]

    def save(self, path):
        state_dict = {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "gamma": self.gamma,
            "epsilon": self.EPSILON,
            "current_episode": self.current_episode,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update": self.TARGET_UPDATE,
            "beta": self.BETA,
        }

        with open(path) as f:
            torch.save(state_dict, f)

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

    def learn(
        self, total_episodes, checkpoint_freq: int = 25_000, checkpoint_path=None
    ):
        self.toggle_train(total_episodes)
        total_steps = 0

        for _ in range(self.current_episode, total_episodes):
            self.env.reset()
            current_step, done = 0, False

            while not done and current_step < self.horizon:
                total_steps += 1
                state = self.env.render()
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.store(
                    Transition(state, reward, action, next_state, done)
                )

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
                    self.replay_memory.update_priorities(
                        indices,
                        priorities.data.detach().squeeze().cpu().numpy().tolist(),
                    )

                    # Back propagation
                    loss = loss.mean()
                    self._back_propagate(loss)

                if total_steps % self.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.controller.state_dict())

            self.current_episode += 1
            if self.current_episode % checkpoint_freq == 0:
                self.save(checkpoint_path)

            self.decrement_epsilon()
            self.increment_beta()

    def increment_beta(self):
        """Increment the beta exponent."""
        self.BETA = min(self.BETA + self.BETA_INCREMENT_CONSTANT, 1)

    def toggle_train(
        self,
        total_episodes: int,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1,
        max_beta: float = 0.4,
    ):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        super().toggle_train(total_episodes, min_epsilon, max_epsilon)

        # PER
        self.REPLAY_CONSTANT = 1e-5
        self.MAX_BETA = max_beta
        self.BETA_INCREMENT_CONSTANT = self.MAX_BETA / (0.75 * self.train_episodes)
