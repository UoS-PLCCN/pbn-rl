import pytest

from .memory import ExperienceReplay, PrioritisedER, Transition


class TestExperienceReplay:
    buffer_size = 10
    buffer = ExperienceReplay(capacity=buffer_size)

    def setup_method(self):
        self.buffer.clear()

    def _gen_experience(self, suffix, reward=1):
        return Transition(
            state=f"state{suffix}",
            action=f"action{suffix}",
            reward=reward,
            next_state=f"next_state{suffix}",
            done=False,
        )

    def _get_buffer(self):
        batch = self.buffer.sample(self.buffer_size)
        batch.sort(key=lambda x: x[3])  # For checking purposes
        return batch

    def test_store(self):
        for i in range(self.buffer_size):
            t = self._gen_experience(f"_{i}", reward=i)
            self.buffer.store(t)
            assert len(self.buffer) == i + 1

        batch = self._get_buffer()
        for i, sample in enumerate(batch):
            state, action, reward, next_state, done = sample
            assert state == f"state_{i}"
            assert action == f"action_{i}"
            assert reward == i
            assert next_state == f"next_state_{i}"
            assert not done

        self.buffer.store(self._gen_experience(f"_11", reward=11))
        assert len(self.buffer) == self.buffer_size  # Ensure loopback

        # Make sure the first element was removed
        batch = self._get_buffer()
        t0 = self._gen_experience(f"_{0}", reward=0)
        for i, sample in enumerate(batch):
            state, action, reward, next_state, done = sample
            assert state != t0.state
            assert action != t0.action
            assert reward != t0.reward
            assert next_state != t0.next_state
            assert not done

    def test_sample(self):
        for i in range(self.buffer_size):
            t = self._gen_experience(f"_{i}", reward=i)
            self.buffer.store(t)
            assert len(self.buffer) == i + 1

        batch = self.buffer.sample(5)
        assert len(batch) == 5


class TestPrioritisedER(TestExperienceReplay):
    buffer_size = 10
    buffer = PrioritisedER(buffer_size)
    beta = 0.4

    def test_sample(self):
        for i in range(self.buffer_size):
            t = self._gen_experience(f"_{i}", reward=i)
            self.buffer.store(t)
            assert len(self.buffer) == i + 1

        batch, indices, weights = self.buffer.sample(5)
        assert len(batch) == 5

    def test_store(self):
        for i in range(self.buffer_size):  # Fill the buffer
            t = self._gen_experience(f"_{i}", reward=i)
            self.buffer.store(t)
            assert len(self.buffer) == i + 1

        # Assert state
        assert self.buffer.max_priority == 1.0
        assert self.buffer.alpha == 0.6
        default_prio = self.buffer.max_priority**self.buffer.alpha

        # Test buffer contents
        samples, indices, weights = self.buffer.sample(self.buffer_size // 2, self.beta)
        for i, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            assert state == f"state_{indices[i]}"
            assert action == f"action_{indices[i]}"
            assert reward == indices[i]
            assert next_state == f"next_state_{indices[i]}"
            assert not done

            # Assert weights
            prob_sum = default_prio * self.buffer_size
            prob = default_prio / prob_sum
            assert weights[i] == (self.buffer.capacity * prob) ** (-self.beta)

    def test_update_priorities(self):
        for i in range(self.buffer_size):  # Fill the buffer
            t = self._gen_experience(f"_{i}", reward=i)
            self.buffer.store(t)
            assert len(self.buffer) == i + 1

        prios = [5, 3, 1, 6, 2]
        indices = [0, 2, 4, 6, 8]

        self.buffer.update_priorities(indices, prios)

        all_prios = [1.0 for _ in range(self.buffer_size)]
        for i, index in enumerate(indices):
            all_prios[index] = prios[i]

        all_probs = [prio**self.buffer.alpha for prio in all_prios]
        prob_sum = sum(all_probs)
        min_prob = min(all_probs) / prob_sum
        max_weight = (self.buffer.capacity * min_prob) ** (-self.beta)

        samples, indices, weights = self.buffer.sample(self.buffer_size // 2, self.beta)
        for i, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            assert state == f"state_{indices[i]}"
            assert action == f"action_{indices[i]}"
            assert reward == indices[i]
            assert next_state == f"next_state_{indices[i]}"
            assert not done

            # Assert weights
            prob = all_probs[indices[i]] / prob_sum
            assert weights[i] == pytest.approx(
                (self.buffer.capacity * prob) ** (-self.beta) / max_weight, 3
            )
