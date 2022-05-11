import gym
import pytest

from . import DDQN, DDQNPER


class TestAgent:
    agent = DDQN
    env_name = "CartPole-v1"
    batch_size = 64

    def setup_method(self):
        self.env = gym.make(self.env_name)
        self.a = self.agent(self.env, batch_size=self.batch_size)

    def test_init(self):
        assert self.a.EPSILON == 1
        assert self.a.num_timesteps == 0
        assert self.a.gamma == pytest.approx(0.99)
        assert self.a.output_size == self.env.action_space.n

    def test_basic_train(self):
        steps = 500
        self.a.learn(steps, log=False)

        assert self.a.EPSILON == pytest.approx(0.01)  # Assert epsilon 0d
        assert self.a.EPSILON_DECREMENT == pytest.approx((1 - 0.01) / steps)

        # Minibatch
        minibatch = self.a._fetch_minibatch()
        batch_zip = list(zip(*minibatch))
        states, actions, rewards, next_states, dones = minibatch
        assert len(batch_zip) == self.batch_size
        assert states[0].size()[0] == self.a.input_size
        assert next_states[0].size()[0] == self.a.input_size
        assert actions[0].size()[0] == 1
        assert rewards[0].size()[0] == 1
        assert type(rewards[0].item()) == float
        assert dones[0].size()[0] == 1
        assert type(dones[0].item()) == float
        assert dones[0].item() == pytest.approx(0.0) or dones[
            0
        ].item() == pytest.approx(1.0)

        # Action selection
        state = states[0]
        q_vals = self.a.controller(state)
        action = q_vals.max(0)[1].item()

        assert self.a._get_learned_action(state) == action

        # Predict
        assert self.a.predict(state, deterministic=True) == action
        self.a.controller.train(False)
        assert self.a.predict(state, deterministic=True) == action
        self.a.controller.train()
        self.a.EPSILON = 0
        assert self.a.predict(state, deterministic=True) == action


class TestPERAgent(TestAgent):
    agent = DDQNPER

    def test_basic_train(self):
        steps = 500
        self.a.learn(steps, log=False)

        assert self.a.EPSILON == pytest.approx(0.01)  # Assert epsilon 0d
        assert self.a.EPSILON_DECREMENT == pytest.approx((1 - 0.01) / steps)

        # Minibatch
        minibatch = self.a._fetch_minibatch()
        states, actions, rewards, next_states, dones = minibatch["experiences"]
        batch_zip = list(zip(*minibatch["experiences"]))
        indices, weights = minibatch["per_data"]
        assert len(batch_zip) == self.batch_size
        assert states[0].size()[0] == self.a.input_size
        assert next_states[0].size()[0] == self.a.input_size
        assert actions[0].size()[0] == 1
        assert rewards[0].size()[0] == 1
        assert type(rewards[0].item()) == float
        assert dones[0].size()[0] == 1
        assert type(dones[0].item()) == float
        assert dones[0].item() == pytest.approx(0.0) or dones[
            0
        ].item() == pytest.approx(1.0)

        # Action selection
        state = states[0]
        q_vals = self.a.controller(state)
        action = q_vals.max(0)[1].item()

        assert self.a._get_learned_action(state) == action

        # Predict
        assert self.a.predict(state, deterministic=True) == action
        self.a.controller.train(False)
        assert self.a.predict(state, deterministic=True) == action
        self.a.controller.train()
        self.a.EPSILON = 0
        assert self.a.predict(state, deterministic=True) == action

        assert self.a.BETA == pytest.approx(1.0)
        assert self.a.BETA_INCREMENT == pytest.approx(0.6 / (0.75 * steps))
