"""
main.py - This module holds the actual Agent.
"""
import random
from math import prod
from pathlib import Path
from statistics import mean

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary
from gym_PBN.envs.pbn_target import PBNTargetEnv
from torch import log_, optim
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
        buffer_size: int = 1_000_000,
        batch_size: int = 32,
        target_update: int = 10_000,
        gamma=0.99,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        exploration_fraction: float = 0.1,
        learning_rate: float = 0.0001,
        max_grad_norm: float = 10.0,
    ):
        self.device = device

        allowed_types = (Discrete, Box, MultiBinary)
        assert (
            type(env.observation_space) in allowed_types
        ), "Only Discrete or Box action space is supported"
        self.input_size = (
            input_size if input_size else prod(env.observation_space.shape)
        )

        assert isinstance(
            env.action_space, Discrete
        ), "Only Discrete action space is supported"
        self.output_size = output_size if output_size else env.action_space.n

        # Get episode stats
        self.env = gym.wrappers.RecordEpisodeStatistics(env)

        # Networks
        self.policy_kwargs = policy_kwargs
        self.controller = DQN(self.input_size, self.output_size, **policy_kwargs).to(
            self.device
        )
        self.target = DQN(self.input_size, self.output_size, **policy_kwargs).to(
            self.device
        )
        self.target.load_state_dict(self.controller.state_dict())
        self.learning_rate = learning_rate

        # Reinforcement learning parameters
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.EPSILON = 1
        self.TARGET_UPDATE = target_update
        self.LOG_INTERVAL = 100
        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.exploration_fraction = exploration_fraction
        self.max_grad_norm = max_grad_norm

        # Memory
        self.replay_memory = ExperienceReplay(self.buffer_size)

        # State
        self.num_timesteps = 0
        self.log = False

    @classmethod
    def _load_helper(cls, path, env, device):
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
            max_epsilon=state_dict["max_epsilon"],
            min_epsilon=state_dict["min_epsilon"],
            exploration_fraction=state_dict["exploration_fraction"],
            learning_rate=state_dict["learning_rate"],
            max_grad_norm=state_dict["max_grad_norm"],
        )
        agent.controller.load_state_dict(state_dict["params"])
        agent.target.load_state_dict(state_dict["params"])
        agent.EPSILON = state_dict["epsilon"]
        agent.num_timesteps = state_dict["num_timesteps"]
        agent.train_steps = state_dict["train_steps"]
        return agent, state_dict

    @classmethod
    def load(cls, path, env: PBNTargetEnv = None, device: torch.device = "cpu"):
        agent, _ = cls._load_helper(path, env, device)
        return agent

    def _make_save_dict(self):
        return {
            "params": self.controller.state_dict(),
            "policy_kwargs": self.policy_kwargs,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "epsilon": self.EPSILON,
            "target_update": self.TARGET_UPDATE,
            "gamma": self.gamma,
            "max_epsilon": self.MAX_EPSILON,
            "min_epsilon": self.MIN_EPSILON,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "train_steps": self.train_steps,
            "num_timesteps": self.num_timesteps,
        }

    def save(self, path):
        state_dict = self._make_save_dict()

        torch.save(state_dict, path / f"ddqn_{self.num_timesteps}.pt")

    def _get_learned_action(self, state) -> int:
        with torch.no_grad():
            q_vals = self.controller(torch.tensor(state).float().to(self.device))
            # max along the 0th dimension, get the index of the max value, return it
            action = q_vals.max(dim=0)[1].item()
        return action

    def predict(self, state, deterministic: bool = False) -> int:
        if not deterministic and random.random() <= self.EPSILON:
            return self.env.action_space.sample()
        else:
            return self._get_learned_action(state)

    def _process_experiences(self, experiences):
        field_order = ("state", "action", "reward", "next_state", "done")
        assert experiences[0]._fields == field_order, "Invalid experiences"

        states, actions, rewards, next_states, dones = zip(*experiences)

        # Load to device
        states = (
            torch.tensor(np.array(states))
            .float()
            .to(self.device)
            .view(self.batch_size, self.input_size)
        )
        actions = torch.tensor(actions).long().to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards).float().to(self.device).unsqueeze(1)
        next_states = (
            torch.tensor(np.array(next_states))
            .float()
            .to(self.device)
            .view(self.batch_size, self.input_size)
        )
        dones = torch.tensor(dones).float().to(self.device).unsqueeze(1)

        return (states, actions, rewards, next_states, dones)

    def _fetch_minibatch(self) -> Minibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            Minibatch: a minibatch.
        """
        # Fetch data
        experiences = self.replay_memory.sample(self.batch_size)
        return self._process_experiences(experiences)

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
            action_prime = self.controller(next_states).max(dim=1)[1].unsqueeze(1)
            target_Q = rewards + (1 - dones) * self.gamma * self.target(
                next_states
            ).gather(1, action_prime)

        # Calculate current and target Q to calculate loss
        controller_Q = self.controller(states).gather(1, actions)
        if self.log:
            self.writer.add_scalar(
                "losses/q_values", controller_Q.mean().item(), self.num_timesteps
            )

        return F.smooth_l1_loss(controller_Q, target_Q, reduction=reduction)

    def _back_propagate(self, loss: torch.Tensor):
        """Do a step of back propagation based on a loss vector.

        Args:
            loss (torch.Tensor): the loss vector as a tensor.
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.log:
            self.wandb.log({"loss": loss})

    def _log_params(self):
        self.writer.add_scalar("rollout/epsilon", self.EPSILON, self.num_timesteps)

    def _learn_step(self):
        batch = self._fetch_minibatch()
        loss = self._get_loss(*batch)
        self._back_propagate(loss)

        if self.log and self.num_timesteps % self.LOG_INTERVAL == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.num_timesteps)

    def _schedule_step(self, step, checkpoint_freq, checkpoint_path=None):
        self.num_timesteps += 1
        self.decrement_epsilon()
        if (step + 1) % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.controller.state_dict())

        if (step + 1) % checkpoint_freq == 0 and checkpoint_path:
            self.save(checkpoint_path)

    def _get_config(self):
        return {
            "gamma": self.gamma,
            "policy": self.policy_kwargs,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "target_update": self.TARGET_UPDATE,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epsilon": self.MAX_EPSILON,
            "min_epsilon": self.MIN_EPSILON,
            "exploration_fraction": self.exploration_fraction,
            "max_grad_norm": self.max_grad_norm,
            "train_steps": self.train_steps,
        }

    def learn(
        self,
        total_steps,
        learning_starts: int = 50_000,
        train_frequency: int = 1,
        checkpoint_freq: int = 25_000,
        checkpoint_path: str = None,
        resume_steps: int = None,
        log: bool = True,
        log_name: str = "ddqn",
        log_dir=None,
        wandb_params={"project": "pbn-rl", "entity": "uos-plccn"},
    ):
        self.toggle_train(total_steps)
        if resume_steps:
            self.num_timesteps = resume_steps

        if log:
            self.log = True
            import wandb

            config = self._get_config()
            config["train_frequency"] = train_frequency
            config["learning_starts"] = learning_starts
            run = wandb.init(
                project=wandb_params["project"],
                entity=wandb_params["entity"],
                sync_tensorboard=True,
                config=config,
                name=log_name,
                save_code=True,
            )
            wandb.watch(self.controller, log="all", log_freq=self.LOG_INTERVAL)
            self.writer = SummaryWriter(Path(log_dir) / log_name)
            self.wandb = wandb
            hyperparam_print = "\n".join(
                ["|param|value|", "|-|-|"]
                + [f"|{key}|{value}" for key, value in config.items()]
            )
            self.writer.add_text("hyperparameters", hyperparam_print)

        episodes = 0

        state = self.env.reset()
        for global_step in range(self.num_timesteps, self.train_steps):
            action = self.predict(state)
            next_state, reward, done, info = self.env.step(action)

            if "episode" in info.keys() and log:
                episodes += 1
                if episodes % self.env.return_queue.maxlen == 0:
                    ep_rew_mean = mean(self.env.return_queue)
                    ep_len_mean = mean(self.env.length_queue)
                    print(
                        f"Episode {episodes}: rew_100 - {ep_rew_mean}, len_100 - {ep_len_mean}"
                    )
                    self.writer.add_scalar(
                        "rollout/ep_rew_mean",
                        ep_rew_mean,
                        self.num_timesteps,
                    )
                    self.writer.add_scalar(
                        "rollout/ep_len_mean",
                        ep_len_mean,
                        self.num_timesteps,
                    )
                    self._log_params()

            self.replay_memory.store(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            if (
                self.num_timesteps > learning_starts
                and len(self.replay_memory) >= self.batch_size
                and global_step % train_frequency == 0
            ):
                self._learn_step()

            # Schedules
            self._schedule_step(global_step, checkpoint_freq, checkpoint_path)

            state = next_state
            if done:
                state = self.env.reset()

        # Cleanup
        self.env.close()
        if log:
            self.writer.close()
            run.finish()

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

    def toggle_train(
        self,
        train_steps: int,
    ):
        """Setting all of the training params."""
        self.controller.train()
        self.train_steps = train_steps

        self.optimizer = optim.Adam(self.controller.parameters(), lr=self.learning_rate)

        # Explore-exploit
        self.EPSILON_DECREMENT = (self.MAX_EPSILON - self.MIN_EPSILON) / (
            self.exploration_fraction * self.train_steps
        )


class DDQNPER(DDQN):
    """Agent using Prioritized Experience Replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BETA = 0.4
        self.MIN_BETA = 0.4
        self.MAX_BETA = 1.0
        self.ALPHA = 0.6
        self.REPLAY_CONSTANT = 1e-5
        self.beta_fraction = 0.75
        self.replay_memory = PrioritisedER(self.buffer_size, self.ALPHA)

    @classmethod
    def load(cls, path, env: PBNTargetEnv = None, device: torch.device = "cpu"):
        agent, state_dict = cls._load_helper(path, env, device)
        agent.BETA = state_dict["beta"]
        return agent

    def _make_save_dict(self):
        ret = super()._make_save_dict()
        ret["beta"] = self.BETA
        return ret

    def _log_params(self):
        super()._log_params()
        self.writer.add_scalar("rollout/beta", self.BETA, self.num_timesteps)

    def _learn_step(self):
        minibatch = self._fetch_minibatch()
        indices, weights = minibatch["per_data"]
        loss = self._get_loss(*minibatch["experiences"], reduction="none")
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

        if self.log and self.num_timesteps % self.LOG_INTERVAL == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.num_timesteps)

    def _schedule_step(self, step, checkpoint_freq, checkpoint_path=None):
        super()._schedule_step(step, checkpoint_freq, checkpoint_path)
        self.increment_beta()

    def _fetch_minibatch(self) -> PERMinibatch:
        """Fetch a minibatch from the replay memory and load it into the chosen device.

        Returns:
            PERMinibatch: a minibatch.
        """
        # Fetch data
        experiences, indices, weights = self.replay_memory.sample(
            self.batch_size, self.BETA
        )
        assert isinstance(indices[0], int), "Invalid indices"
        assert isinstance(weights[0], np.float32), "Invalid weights"

        processed_experiences = self._process_experiences(experiences)
        weights = torch.tensor(weights).float().to(self.device).squeeze().unsqueeze(1)

        return {
            "experiences": processed_experiences,
            "per_data": (indices, weights),
        }

    def _get_config(self):
        config = super()._get_config()
        config["max_beta"] = self.MAX_BETA
        config["min_beta"] = self.MIN_BETA
        config["beta_fraction"] = self.beta_fraction
        config["per_alpha"] = self.ALPHA
        return config

    def increment_beta(self):
        """Increment the beta exponent."""
        self.BETA = min(self.BETA + self.BETA_INCREMENT, 1)

    def toggle_train(
        self,
        train_steps: int,
    ):
        """Setting all of the training params.

        Args:
            conf (TrainingConfig): the training configuration
        """
        super().toggle_train(train_steps)

        # Reach 1 after 75% of training
        self.BETA_INCREMENT = (self.MAX_BETA - self.MIN_BETA) / (
            self.beta_fraction * self.train_steps
        )
