"""
main.py - This module holds the actual Agent.
"""
import random
from collections import deque
from math import prod

import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary
from gym_PBN.envs.pbn_target import PBNTargetEnv
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .memory import ExperienceReplay, PrioritisedER, Transition
from .network import DQN
from .types import Minibatch, PERMinibatch


class DDQN:
    """The agent of the RL algorithm. Houses the DQN, ER, etc."""

    def __init__(
        self,
        env: PBNTargetEnv = None,
        device: torch.device = "cpu",
        input_size: int = None,
        output_size: int = None,
        policy_kwargs: dict = {"net_arch": [(64, 64)]},
        buffer_size: int = 50_000,
        batch_size: int = 256,
        target_update: int = 5000,
        gamma=0.99,
    ):
        self.device = device

        self.input_size = input_size
        if not self.input_size:
            if type(env.observation_space) == MultiBinary:
                self.input_size = env.observation_space.n
            elif type(env.observation_space) == Box:
                self.input_size = prod(env.observation_space.shape)

        # TODO Only discrete fro now
        self.output_size = output_size if output_size else env.action_space.n

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
        self.LOG_INTERVAL = 100

        # Memory
        self.replay_memory = ExperienceReplay(self.buffer_size)

        # State
        self.num_timesteps = 0
        self.num_episodes = 0

    @classmethod
    def load(cls, path, env: PBNTargetEnv = None, device: torch.device = "cpu"):
        state_dict = torch.load(path)
        agent = cls(
            env,
            device,
            gamma=state_dict["gamma"],
            policy_kwargs=state_dict["policy_kwargs"],
            input_size=state_dict["input_size"],
            output_size=state_dict["output_size"],
            buffer_size=state_dict["buffer_size"],
            batch_size=state_dict["batch_size"],
            target_update=state_dict["target_update"],
        )
        agent.controller.load_state_dict(state_dict["params"])
        agent.target.load_state_dict(state_dict["params"])
        agent.EPSILON = state_dict["epsilon"]
        agent.num_timesteps = state_dict["num_timesteps"]
        agent.num_episodes = state_dict["num_episodes"]
        agent.train_steps = state_dict["train_steps"]
        return agent

    def save(self, path):
        state_dict = {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "epsilon": self.EPSILON,
            "target_update": self.TARGET_UPDATE,
            "train_steps": self.train_steps,
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
        }

        with open(path / f"ddqn_{self.num_timesteps}.pt") as f:
            torch.save(state_dict, f)

    def _get_learned_action(self, state) -> int:
        with torch.no_grad():
            q_vals = self.controller(
                torch.tensor(state, device=self.device, dtype=torch.float)
            )
            # max along the 0th dimension, get the index of the max value, return it
            action = q_vals.max(0)[1].item()
        return action

    def predict(self, state, deterministic: bool = False) -> int:
        if (
            self.controller.training
            and not deterministic
            and random.uniform(0, 1) <= self.EPSILON
        ):
            return self.env.action_space.sample()
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
        self,
        total_steps,
        checkpoint_freq: int = 25_000,
        checkpoint_path=None,
        resume_steps: int = None,
        log=True,
        log_dir=None,
    ):
        if log:
            writer = SummaryWriter(log_dir)

        self.toggle_train(total_steps)
        training_done = False
        if resume_steps:
            self.num_timesteps = resume_steps

        episode_reward, episode_steps = 0, 0

        while not training_done:
            state = self.env.reset()
            done = False

            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.store(
                    Transition(state, action, reward, next_state, done)
                )
                episode_reward += reward

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

                self.num_timesteps += 1
                episode_steps += 1

                if self.num_timesteps % self.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.controller.state_dict())

                if self.num_timesteps % checkpoint_freq == 0 and checkpoint_path:
                    self.save(checkpoint_path)

                self.decrement_epsilon()
                if log:
                    writer.add_scalar(
                        "hyperparams/epsilon", self.EPSILON, self.num_timesteps
                    )

                state = next_state

                if self.num_timesteps == self.train_steps:
                    training_done = True
                    break

            self.num_episodes += 1
            if self.num_episodes % self.LOG_INTERVAL == 0 and log:
                writer.add_scalar(
                    "rollout/ep_rew_mean",
                    episode_reward / self.LOG_INTERVAL,
                    self.num_timesteps,
                )
                writer.add_scalar(
                    "rollout/ep_len_mean",
                    episode_steps / self.LOG_INTERVAL,
                    self.num_timesteps,
                )
                episode_reward = 0
                episode_steps = 0

        self.env.close()

        if log:
            writer.close()

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    def toggle_train(
        self,
        train_steps: int,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1,
    ):
        """Setting all of the training params."""
        self.controller.train()
        self.target.train()
        self.train_steps = train_steps

        self.optimizer = optim.RMSprop(self.controller.parameters(), lr=0.01)

        # Explore-exploit
        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.EPSILON_DECREMENT = (
            self.MAX_EPSILON - self.MIN_EPSILON
        ) / self.train_steps


class DDQNPER(DDQN):
    """Agent using Prioritized Experience Replay."""

    def __init__(
        self,
        env: PBNTargetEnv = None,
        device: torch.device = "cpu",
        input_size: int = None,
        output_size: int = None,
        policy_kwargs: dict = {"net_arch": [(64, 64)]},
        buffer_size: int = 50_000,
        batch_size: int = 256,
        target_update: int = 5000,
        gamma=0.99,
    ):
        super().__init__(
            env,
            device,
            input_size,
            output_size,
            policy_kwargs,
            buffer_size,
            batch_size,
            target_update,
            gamma,
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
            input_size=state_dict["input_size"],
            output_size=state_dict["output_size"],
            buffer_size=state_dict["buffer_size"],
            batch_size=state_dict["batch_size"],
            target_update=state_dict["target_update"],
        )
        agent.controller.load_state_dict(state_dict["params"])
        agent.target.load_state_dict(state_dict["params"])
        agent.EPSILON = state_dict["epsilon"]
        agent.BETA = state_dict["beta"]
        agent.num_timesteps = state_dict["num_timesteps"]
        agent.num_episodes = state_dict["num_episodes"]
        agent.train_steps = state_dict["train_steps"]
        return agent

    def save(self, path):
        state_dict = {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "epsilon": self.EPSILON,
            "target_update": self.TARGET_UPDATE,
            "beta": self.BETA,
            "train_steps": self.train_steps,
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
        }

        torch.save(state_dict, path / f"ddqnper_{self.num_timesteps}.pt")

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
            np.array(state_batch), device=self.device, dtype=torch.float
        ).view(self.batch_size, self.input_size)
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.array(next_state_batch), device=self.device, dtype=torch.float
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
        self,
        total_steps,
        checkpoint_freq: int = 25_000,
        checkpoint_path=None,
        resume_steps: int = None,
        log=True,
        log_dir=None,
    ):
        if log:
            writer = SummaryWriter(log_dir)

        self.toggle_train(total_steps)
        training_done = False
        if resume_steps:
            self.num_timesteps = resume_steps

        episode_rewards, episode_steps = deque(maxlen=self.LOG_INTERVAL), deque(
            maxlen=self.LOG_INTERVAL
        )

        while not training_done:
            state = self.env.reset()
            done = False
            episode_reward, episode_step = 0, 0

            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.store(
                    Transition(state, action, reward, next_state, done)
                )
                episode_reward += reward

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
                        priorities.data.detach().squeeze().abs().cpu().numpy().tolist(),
                    )

                    # Back propagation
                    loss = loss.mean()
                    self._back_propagate(loss)

                self.num_timesteps += 1
                episode_step += 1

                if self.num_timesteps % self.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.controller.state_dict())

                if self.num_timesteps % checkpoint_freq == 0:
                    self.save(checkpoint_path)

                self.decrement_epsilon()
                self.increment_beta()
                if log:
                    writer.add_scalar("hyperparams/beta", self.BETA, self.num_timesteps)
                    writer.add_scalar(
                        "hyperparams/epsilon", self.EPSILON, self.num_timesteps
                    )

                state = next_state

                if self.num_timesteps == self.train_steps:
                    training_done = True
                    break

            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            self.num_episodes += 1
            if self.num_episodes % self.LOG_INTERVAL == 0:
                if log:
                    writer.add_scalar(
                        "rollout/ep_rew_mean",
                        sum(episode_rewards) / self.LOG_INTERVAL,
                        self.num_timesteps,
                    )
                    writer.add_scalar(
                        "rollout/ep_len_mean",
                        sum(episode_steps) / self.LOG_INTERVAL,
                        self.num_timesteps,
                    )

        self.env.close()
        if log:
            writer.close()

    def increment_beta(self):
        """Increment the beta exponent."""
        self.BETA = min(self.BETA + self.BETA_INCREMENT, 1)

    def toggle_train(
        self,
        train_steps: int,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1,
        max_beta: float = 1.0,
    ):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        super().toggle_train(train_steps, min_epsilon, max_epsilon)

        # PER
        self.REPLAY_CONSTANT = 1e-5
        self.MIN_BETA = 0.4
        self.MAX_BETA = max_beta
        # Reach 1 after 75% of training
        self.BETA_INCREMENT = (self.MAX_BETA - self.MIN_BETA) / (
            0.75 * self.train_steps
        )
