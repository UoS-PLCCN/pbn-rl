from tqdm import tqdm
import numpy as np
import itertools
from pathlib import Path
import plotly.express as px
import pandas as pd


def ssd_eval(env, model=None, output=None):
    SSD_N = 1_200_000  # Number of experiences to sample for the SSD calculatiom
    SSD_RESETS = 300
    BIT_FLIP_PROB = 0

    g = len(env.target_nodes)
    ssd = np.zeros((2**g, SSD_RESETS), dtype=np.float32)

    total_iters = 0
    for i in tqdm(range(SSD_RESETS), desc=f"SSD run for {env.name}"):
        sub_ssd = np.zeros(2**g, dtype=np.float32)
        env.reset()

        for _ in range(SSD_N // SSD_RESETS):
            total_iters += 1

            state = env.render()
            # Convert relevant part of state to binary string, then parse it as an int to get the bucket index.
            bucket = env.render(mode="target_idx")
            sub_ssd[bucket] += 1

            if not model:  # Control the environment
                flip = np.random.rand(len(state)) < BIT_FLIP_PROB
                for j in range(len(state)):
                    if flip[j]:
                        env.graph.flipNode(j)
                env.step(action=0)
            else:
                action, _ = model.predict(state, deterministic=True)
                env.step(action=action)

        ssd[:, i] = sub_ssd

    ssd = np.mean(ssd, axis=1)
    ssd /= SSD_N // SSD_RESETS  # Normalize
    ret = ssd

    to_str = lambda perm: "".join([str(i) for i in perm])
    states = list(map(to_str, itertools.product([0, 1], repeat=g)))
    ret = pd.DataFrame({"states": states, "ssd_values": list(ssd)})

    if output is not None:
        visualize_ssd(ret, output, env)

    return ret


def visualize_ssd(ssd_frame, output, env):
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    fig = px.bar(
        ssd_frame,
        x="states",
        y="ssd_values",
        labels={
            "states": "Gene Premutations",
            "ssd_values": "Steady State Distribution",
        },
        title=f"SSD for {env.name}",
    )
    fig.write_image(output)
