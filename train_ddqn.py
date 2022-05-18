import argparse
import pickle
import random
import time
from pathlib import Path

import gym
import numpy as np
import torch

from ddqn_per import DDQNPER
from eval import compute_ssd_hist

model_cls = DDQNPER
model_name = "DDQNPER"

# Parse settings
parser = argparse.ArgumentParser(description="Train an RL model for target control.")
parser.add_argument(
    "--time-steps", metavar="N", type=int, help="Total number of training time steps."
)
parser.add_argument(
    "--learning-starts", type=int, metavar="LS", help="when the learning starts"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)."
)
parser.add_argument("--env", type=str, help="the environment to run.")
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
parser.add_argument(
    "--exp-name", type=str, default="ddqn", metavar="E", help="the experiment name."
)
parser.add_argument("--log-dir", default="logs", help="path to save logs")
parser.add_argument(
    "--hyperparams", type=str, help="any extra hyper parameters for the model"
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
env = gym.make(args.env)

# set up logs
TOP_LEVEL_LOG_DIR = Path(args.log_dir)
TOP_LEVEL_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"{args.env.split('/')[-1]}_{args.exp_name}_{args.seed}"

# Checkpoints
checkpoint_path = Path(args.checkpoint_dir) / RUN_NAME
checkpoint_path.mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint():
    files = list(checkpoint_path.glob("*.pt"))
    if len(files) > 0:
        return max(files, key=lambda x: x.stat().st_ctime)
    else:
        return None


# Model
total_time_steps = args.time_steps
resume_steps = None
hyperparams = {}
if args.hyperparams:
    hyperparams = {
        param.split("=")[0]: eval(param.split("=")[1])
        for param in args.hyperparams.split(",")
    }
model = DDQNPER(env, DEVICE, **hyperparams)
resume_steps = 0
if args.resume_training:
    model_path = get_latest_checkpoint()

    if model_path:
        print(f"Loading model {model_path}.")
        model = model_cls.load(model_path, env, device=DEVICE)
        resume_steps = total_time_steps - model.num_timesteps


if not args.eval_only:
    print(f"Training for {total_time_steps - resume_steps} time steps...")
    model.learn(
        total_time_steps,
        learning_starts=args.learning_starts,
        checkpoint_freq=10_000,
        checkpoint_path=checkpoint_path,
        resume_steps=resume_steps,
        log_dir=TOP_LEVEL_LOG_DIR,
        log_name=RUN_NAME,
        log=True,
    )


compute_ssd_hist(
    env, model, TOP_LEVEL_LOG_DIR / f"{RUN_NAME}.png", resets=300, iters=100_000
)
