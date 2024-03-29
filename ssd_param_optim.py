import argparse
import random

import gymnasium as gym
import gym_PBN
import numpy as np
from gym_PBN.utils.eval import compute_ssd_hist

# Parse settings
parser = argparse.ArgumentParser(description="Evaluate SSD computation parameters.")

parser.add_argument(
    "--resets", type=int, default=300, metavar="R", help="number of environment resets."
)
parser.add_argument(
    "--iters",
    type=int,
    default=1_200_000,
    metavar="N",
    help="total number of environment steps.",
)
parser.add_argument("--runs", type=int, default=5, help="total number of SSD runs.")
parser.add_argument(
    "--bit-flip-prob", type=float, default=0.01, help="the probability for bit flips"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)."
)
parser.add_argument("--env", type=str, help="the environment file to run.")
args = parser.parse_args()

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# Load env
env = gym.make(args.env)

results = np.zeros((args.runs, 2 ** len(env.target_nodes)))
for i in range(args.runs):
    ssd = compute_ssd_hist(
        env, iters=args.iters, resets=args.resets, bit_flip_prob=args.bit_flip_prob
    )
    for state, row in ssd.iterrows():
        results[i, int(state, 2)] = row["Value"]

print("Mean:", results.mean(axis=0).mean())
print("Std:", results.std(axis=0).mean())
