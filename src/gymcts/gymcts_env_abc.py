from typing import TypeVar, Any, SupportsFloat, Callable
from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np


class GymctsABC(ABC, gym.Env):

    @abstractmethod
    def get_state(self) -> Any:
        """
        Returns the current state of the environment. The state can be any datatype in principle, that allows to restore
        the environment to the same state. The state is used to restore the environment unsing the load_state method.

        It's recommended to use a numpy array if possible, as it is easy to serialize and deserialize.

        :return: the current state of the environment
        """
        pass

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """
        Loads the state of the environment. The state can be any datatype in principle, that allows to restore the
        environment to the same state. The state is used to restore the environment unsing the load_state method.

        :param state: the state to load
        :return: None
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the environment is in a terminal state, False otherwise.
        :return:
        """
        pass

    @abstractmethod
    def get_valid_actions(self) -> list[int]:
        """
        Returns a list of valid actions for the current state of the environment.
        This used to obtain potential actions/subsequent sates for the MCTS tree.
        :return: the list of valid actions
        """
        pass

    @abstractmethod
    def action_masks(self) -> np.ndarray | None:
        """
        Returns a numpy array of action masks for the environment. The array should have the same length as the number
        of actions in the action space. If an action is valid, the corresponding mask value should be 1, otherwise 0.
        If no action mask is available, it should return None.

        :return: a numpy array of action masks or None
        """
        pass

    @abstractmethod
    def rollout(self) -> float:
        """
        Performs a rollout from the current state of the environment and returns the return (sum of rewards) of the rollout.

        Please make sure the return value is in the interval [-1, 1].
        Otherwise, the MCTS algorithm will not work as expected (due to a male-fitted exploration coefficient;
        exploration and exploitation are not well-balanced then).

        :return: the return of the rollout
        """
        pass
