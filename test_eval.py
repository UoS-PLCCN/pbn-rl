import gym
import gym_PBN
import pytest

from eval import compute_ssd_hist


@pytest.fixture
def env():
    return gym.make("gym-PBN/Bittner-28-v0")


def test_ssd(env):
    for i in range(5):
        ssd = compute_ssd_hist(env, resets=300, iters=100_000, bit_flip_prob=0.01)
        print(f"{i} - 0: {ssd.loc['0', 'Value']}, 1: {ssd.loc['1', 'Value']}")


def test_default_ssd(env):
    compute_ssd_hist(
        env,
        output="logs/uncontrolled.png",
        resets=300,
        iters=100_000,
        bit_flip_prob=0.01,
    )


def test_env(env):
    state = env.reset()
    for step in range(15):
        next_state, reward, done, info = env.step(action=0)
        print(state, next_state, reward, done, info)

        if done:
            break
