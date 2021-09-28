import abc
from typing import Tuple


class BaseEnvironment(abc.ABC):
    """Base class for RL environments.

    Abstract methods:
        step: takes action produces reward and next state.
        reset_environment: reset environment and return initial state.
    """

    def __init__(self):
        self._training: bool
        self._active: bool
        self._episode_step_count: int

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent."""
        pass

    @abc.abstractmethod
    def reset_environment(self, train: bool):
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        pass

    @property
    def active(self) -> bool:
        return self._active

    @property
    def episode_step_count(self) -> int:
        return self._episode_step_count
