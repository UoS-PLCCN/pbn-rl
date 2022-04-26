import itertools
from pathlib import Path
from typing import Union, Iterable

import gym
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from gym_PBN.envs.pbn_target import PBNTargetEnv


def _bit_seq_to_str(seq: Iterable[int]) -> str:
    return "".join([str(i) for i in seq])


def compute_ssd_hist(
    env: PBNTargetEnv,
    model: object = None,
    output: Union[str, Path] = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
) -> pd.DataFrame:
    """Compute a Steady State Distribution Histogram for a given environment,
    perhaps under the control of a given model.

    Args:
        env (PBNTargetEnv): a gym-PBN environment.
        model (object, optional): a Stable Baselines model or one of a similar interface. Defaults to None.
        output (str, optional): a path to output a visualization. Defaults to None.
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
                action, _ = model.predict(state, deterministic=True)
                env.step(action=action)

        ssd[:, i] = sub_ssd

    ssd = np.mean(ssd, axis=1)
    ssd /= SSD_N // SSD_RESETS  # Normalize
    ret = ssd

    states = list(map(_bit_seq_to_str, itertools.product([0, 1], repeat=g)))
    ret = pd.DataFrame(list(ssd), index=states, columns=["Value"])

    if output is not None:
        visualize_ssd(ret, output, env.name)

    return ret


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


def visualize_ssd(ssd_frame: pd.DataFrame, output: Union[str, Path], env_name: str):
    """Visualize and save the Steady State Distribution histogram.

    Args:
        ssd_frame (pd.DataFrame): a DataFrame containing the states of interest and
            their corresponding probability.
        output (Union[str, Path]): a path to save the visualized figure.
        env_name (str): the name of the environment for metadata's sake.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    fig = px.bar(
        ssd_frame,
        x=ssd_frame.index,
        y="Values",
        labels={
            "states": "Gene Premutations",
            "ssd_values": "Steady State Distribution",
        },
        title=f"SSD for {env_name}",
    )
    fig.write_image(output)
