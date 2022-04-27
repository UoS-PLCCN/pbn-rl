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
