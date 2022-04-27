import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import argparse
import torch
import random
import numpy as np
from utils import SaveOnBestTrainingRewardCallback
from pathlib import Path
import json

model = PPO

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

# Load env
with open(args.env, "rb") as f:
    env = pickle.load(f)


# set up logs
logger = configure("logs", ["csv"])
save_callback = SaveOnBestTrainingRewardCallback(check_freq=1_000, log_dir="logs")

# Checkpoints
Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint():
    model_path = Path(args.checkpoint_dir) / "model"
    metadata_path = Path(args.checkpoint_dir) / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    return model_path, metadata


# Model
time_steps = args.time_steps
if args.resume_training:
    model_path, metadata = get_latest_checkpoint()
    model = model.load(model_path, env, device=DEVICE, logger=logger)
    start_step = args.time_steps - metadata["step"]
else:
    model = model("MlpPolicy", env)
    model.logger(logger)

# Train
model.learn(time_steps, callback=SaveOnBestTrainingRewardCallback)
