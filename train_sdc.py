import random
from pathlib import Path

import gym_PBN
import gymnasium as gym
import numpy as np
import torch

from ddqn_per import DDQNPER

# Meta Parameters
SEED = 42
EXP_NAME = "1"
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Training on {DEVICE}")

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Create an SDC env
env = gym.make(
    "gym-PBN/PBCN-sampled-data-v0",
    logic_func_data=(
        ["u1", "u2", "u3", "u4", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        [
            [],
            [],
            [],
            [],  # Control nodes don't have functions
            [("(u1 and x8)", 1)],  # x1
            [("(u2 and x9) or u1", 1)],  # x2
            [("(u2 or x1) and u3", 0.8), ("x3", 0.2)],  # x3
            [("not u4", 1)],  # x4
            [("x7", 0.8), ("not x7", 0.2)],  # x5
            [("(not x3) or u3", 0.7), ("x6", 0.3)],  # x6
            [("x3 and x4 and not x6", 0.8), ("x3 and x4 and x6", 0.2)],  # x7
            [("x3 and x4", 1)],  # x8
            [("x8", 1)],  # x9
        ],
    ),
    goal_config={
        "all_attractors": [{(0, 1, 1, 1, 1, 0, 1, 1, 1)}],
        "target": [{(0, 1, 1, 1, 1, 0, 1, 1, 1)}],
    },
    reward_config={
        "successful_reward": 5,
        "wrong_attractor_cost": 2,
        "action_cost": 1,
    },
    T=10,
    name="PBCN_9-4",
    max_episode_steps=11,  # The horizon from the paper
)

# Model
# NOTE Look in the DDQN code for HACK notes that make the combinatorial action space work.
hyperparams = {  # From the paper
    "gamma": 0.99,
    "min_epsilon": 0.01,
    "beta": 0.4,
    "max_beta": 1,
    "policy_kwargs": {"net_arch": [(50, 50, 50)]},
    "output_size": env.discrete_action_space.n,
}
model = DDQNPER(env, DEVICE, **hyperparams)

# Logs & Checkpoints
TOP_LEVEL_LOG_DIR = Path("logs")
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME = f"{env.name}_{EXP_NAME}_{SEED}"
checkpoint_path = Path("models") / RUN_NAME
checkpoint_path.mkdir(parents=True, exist_ok=True)

# Train
# NOTE This was episodes in the paper. 250k episodes was a legacy thing I just didn't bother playing with at the time, in later work it turned out it was way too much, and also training with a target of # timesteps not # episodes is better practice.w.
total_time_steps = 250_000

print(f"Training for {total_time_steps} time steps...")
model.learn(
    total_time_steps,
    # NOTE In the paper we started training as soon as we had a batch of data. It turns out this is also not the best, but I didn't know that at the time.
    learning_starts=model.get_config()["batch_size"],
    checkpoint_freq=10_000,
    checkpoint_path=checkpoint_path,
    log=True,
    log_dir=TOP_LEVEL_LOG_DIR,
    log_name=RUN_NAME,
)

env.close()
