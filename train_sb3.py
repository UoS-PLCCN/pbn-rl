import argparse
import random
from pathlib import Path

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

model_cls = PPO
model_name = "PPO"

# Parse settings
parser = argparse.ArgumentParser(description="Train an RL model for target control.")
parser.add_argument(
    "--time-steps", metavar="N", type=int, help="Total number of training time steps."
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)."
)
parser.add_argument("--env", type=str, help="the environment file to run.")
parser.add_argument("--env-name", type=str, help="the name of the environment")
parser.add_argument(
    "--resume-training",
    action="store_true",
    help="resume training from latest checkpoint.",
)
parser.add_argument("--checkpoint-dir", default="models", help="path to save models")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--eval-only", action="store_true", default=False, help="evaluate only"
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
print(f"Training on {DEVICE}")

# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load env
# with open(args.env, "rb") as f:
#     env = pickle.load(f)
env = gym.make("CartPole-v1")

# set up logs
TOP_LEVEL_LOG_DIR = Path("logs")
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"{model_name}_{args.env_name}"

# Checkpoints
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
save_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=Path(args.checkpoint_dir) / RUN_NAME,
    name_prefix=RUN_NAME,
)


def get_latest_checkpoint():
    model_checkpoints = Path(args.checkpoint_dir) / RUN_NAME

    files = model_checkpoints.glob("*.zip")
    return max(files, key=lambda x: x.stat().st_ctime)


# Model
time_steps = args.time_steps
model = model_cls(
    "MlpPolicy", env, device=DEVICE, tensorboard_log=TOP_LEVEL_LOG_DIR / RUN_NAME
)

if args.resume_training:
    model_path = get_latest_checkpoint()
    model = model_cls.load(model_path, env, device=DEVICE)

    total_time_steps = args.time_steps
    time_steps = total_time_steps - model.num_timesteps

# Train
if not args.eval_only:
    print(f"Training for {time_steps} time steps...")
    model.learn(
        time_steps,
        tb_log_name=f"run_{time_steps}",
        callback=save_callback,
        reset_num_timesteps=not args.resume_training,
    )
