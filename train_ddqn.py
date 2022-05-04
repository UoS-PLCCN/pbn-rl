from ddqn_per import DDQNPER
import argparse
import torch
import pickle
import numpy as np
import random
from pathlib import Path

# Parse settings
parser = argparse.ArgumentParser(description="Train an RL model for target control.")
parser.add_argument(
    "--time-steps", metavar="N", type=int, help="Total number of training steps."
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)."
)
parser.add_argument("--env", type=str, help="the environment file to run.")
parser.add_argument(
    "--resume-training",
    action="store_true",
    help="resume training from latest checkpoint.",
)
parser.add_argument("--checkpoint-dir", help="path to save models")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Checkpoints
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
checkpoint_path = Path(args.checkpoint_dir) / f"ddqn_per_{Path(args.env).name}.pt"

# Load env
with open(args.env, "rb") as f:
    env = pickle.load(f)

if args.resume_training:
    model = DDQNPER.load(checkpoint_path, env, DEVICE)
else:
    model = DDQNPER(env, DEVICE)

model.learn(250_000, checkpoint_path=checkpoint_path)
