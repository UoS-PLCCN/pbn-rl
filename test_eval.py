import pickle
import pytest
from eval import compute_ssd_hist


@pytest.fixture
def env():
    with open("envs/n_28.pkl", "rb") as f:
        return pickle.load(f)


def test_ssd(env):
    ssd = compute_ssd_hist(env, resets=100, iters=100_000, bit_flip_prob=0)
    print(f"0: {ssd.loc['0', 'Value']}, 1: {ssd.loc['1', 'Value']}")


def test_parameter_ablation_study(env):
    resets = [10, 100, 300, 500]
    iters = [120_000, 510_000, 1_200_000, 1_800_000]
    bit_flip_probability = [0, 0.01, 0.1, 0.3]
    ...
