from typing import List, Tuple

import numpy as np
from key_door import base_environment


class Wrapper(base_environment.BaseEnvironment):
    def __init__(self, env):
        self._env = env

    def step(self, action) -> Tuple[float, Tuple[int, int]]:
        return self._env.step(action)

    def reset_environment(self, train: bool = True) -> Tuple[int, int, int]:
        return self._env.reset_environment(train=train)

    @property
    def active(self) -> bool:
        return self._env.active

    @property
    def episode_step_count(self) -> int:
        return self._env._episode_step_count

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._env._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self._env.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._env._state_space

    @property
    def positional_state_space(self):
        return self._env._positional_state_space

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._env._visitation_counts

    @property
    def train_episode_history(self) -> List[np.ndarray]:
        return self._env._train_episode_history

    @property
    def test_episode_history(self) -> List[np.ndarray]:
        return self._env._test_episode_history

    @property
    def train_episode_position_history(self) -> np.ndarray:
        return np.array(self._env._train_episode_position_history)

    @property
    def test_episode_position_history(self) -> np.ndarray:
        return np.array(self._env._test_episode_position_history)