import itertools
from pathlib import Path
from typing import Iterable, Union

import gym
import numpy as np
import pandas as pd
import plotly.express as px
from gym_PBN.envs.pbn_target import PBNTargetEnv
from tqdm import tqdm


def _bit_seq_to_str(seq: Iterable[int]) -> str:
    return "".join([str(i) for i in seq])


def compute_ssd_hist(
    env: PBNTargetEnv,
    model: object = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
) -> pd.DataFrame:
    """Compute a Steady State Distribution Histogram for a given environment,
    perhaps under the control of a given model.

    Args:
        env (PBNTargetEnv): a gym-PBN environment.
        model (object, optional): a Stable Baselines model or one of a similar interface. Defaults to None.
        iters (int, optional): how many environment transitions to compute. Defaults to 1.2 Million.
        resets (int, optional): how many times to reset the environment. `iters` should be divisible by this number. Defaults to 300.
        bit_flip_prob (float, optional): number in [0,1] on the probability of flipping each bit at random when no control is being applied. Defaults to 0.01

    Returns:
        pd.DataFrame: the Steady State Distribution histogram as a pandas DataFrame.
    """
    SSD_N = iters  # Number of experiences to sample for the SSD calculation
    SSD_RESETS = resets
    BIT_FLIP_PROB = bit_flip_prob

    assert (
        bit_flip_prob >= 0 and bit_flip_prob <= 1
    ), "Invalid Bit Flip Probability value."
    assert SSD_RESETS > 0, "Invalid resets value."
    assert SSD_N > 0, "Invalid iterations value."
    assert SSD_N // SSD_RESETS, "Resets does not divide the iterations."

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
                action = model.predict(state, deterministic=True)
                if type(action) == tuple:
                    action = action[0]
                env.step(action=action)

        ssd[:, i] = sub_ssd

    ssd = np.mean(ssd, axis=1)
    ssd /= SSD_N // SSD_RESETS  # Normalize
    ret = ssd

    states = list(map(_bit_seq_to_str, itertools.product([0, 1], repeat=g)))
    ret = pd.DataFrame(list(ssd), index=states, columns=["Value"])
    plot = visualize_ssd(ret, env.name)

    return ret, plot


def eval_increase(
    env: PBNTargetEnv,
    model: object,
    original_ssd: pd.DataFrame = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
) -> float:
    """Compute the total increase in the favourable states in the SSD histogram.

    Args:
        env (PBNTargetEnv): the gym-PBN environment.
        model (object): a Stable Baselines model or an Agent of a similar interface.
        original_ssd (pd.DataFrame, optional): the cached uncontrolled SSD Histogram. Defaults to None,
            and if not provided, it will be recalculated.
        iters (int, optional): how many environment transitions to compute. Defaults to 1.2 Million.
        resets (int, optional): how many times to reset the environment. `iters` should be divisible by this number. Defaults to 300.
        bit_flip_prob (float, optional): number in [0,1] on the probability of flipping each bit at random when no control is being applied. Defaults to 0.01

    Returns:
        float: the total increase across all favourable states.
    """
    if original_ssd == None:  # Cache
        original_ssd = compute_ssd_hist(
            env, iters=iters, resets=resets, bit_flip_prob=bit_flip_prob
        )
    model_ssd = compute_ssd_hist(
        env, model, iters=iters, resets=resets, bit_flip_prob=bit_flip_prob
    )
    states_of_interest = [_bit_seq_to_str(state) for state in env.target_node_values]
    return (model_ssd - original_ssd)[states_of_interest].sum()


def visualize_ssd(ssd_frame: pd.DataFrame, env_name: str) -> object:
    """Visualize and save the Steady State Distribution histogram.

    Args:
        ssd_frame (pd.DataFrame): a DataFrame containing the states of interest and
            their corresponding probability.
        env_name (str): the name of the environment for metadata's sake.
    """
    fig = px.bar(
        ssd_frame,
        x=ssd_frame.index,
        y="Value",
        labels={
            "states": "Gene Premutations",
            "ssd_values": "Steady State Distribution",
        },
        title=f"SSD for {env_name}",
    )
    return fig
