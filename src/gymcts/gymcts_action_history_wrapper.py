import random

import numpy as np
from typing import Any, SupportsFloat, Callable
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from gymcts.gymcts_env_abc import GymctsABC

from gymcts.logger import log


class ActionHistoryMCTSGymEnvWrapper(GymctsABC, gym.Wrapper):
    """
    A wrapper for gym environments that implements the GymctsABC interface.
    It uses the action history as state representation.
    Please note that this is not the most efficient way to implement the state representation.
    It is supposed to be used to see if your use-case works well with the MCTS algorithm.
    If it does, you can consider implementing all GymctsABC methods in a more efficient way.
    The action history is a list of actions taken in the environment.
    The state is represented as a list of actions taken in the environment.
    The state is used to restore the environment using the load_state method.

    It is supposed to be used to see if your use-case works well with the MCTS algorithm.
    If it does, you can consider implementing all GymctsABC methods in a more efficient way.
    """

    # helper attributes for the wrapper
    _terminal_flag: bool = False
    _last_reward: SupportsFloat = 0
    _step_tuple: tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]] = None

    _action_mask_fn: Callable[[gym.Env], np.ndarray] | None = None

    def __init__(
            self,
            env,
            action_mask_fn: str | Callable[[gym.Env], np.ndarray] | None = None,
            buffer_length: int = 100,
    ):
        """
        A wrapper for gym environments that implements the GymctsABC interface.
        It uses the action history as state representation.
        Please note that this is not the most efficient way to implement the state representation.
        It is supposed to be used to see if your use-case works well with the MCTS algorithm.
        If it does, you can consider implementing all GymctsABC methods in a more efficient way.

        :param env: the environment to wrap
        :param action_mask_fn: a function that takes the environment as input and returns a mask of valid actions
        :param buffer_length: the length of the buffer for recording episodes for determining their rollout returns
        """
        # wrap with RecordEpisodeStatistics if it is not already wrapped
        env = RecordEpisodeStatistics(env, buffer_length=buffer_length)

        gym.Wrapper.__init__(self, env)

        self._wrapper_action_history = []

        # assert that the action space is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Only discrete action spaces are supported.")

        if action_mask_fn is not None:
            # copy of stable baselines3 contrib implementation
            if isinstance(action_mask_fn, str):
                found_method = getattr(self.env, action_mask_fn)
                if not callable(found_method):
                    raise ValueError(f"Environment attribute {action_mask_fn} is not a method")

                self._action_mask_fn = found_method
            else:
                self._action_mask_fn = action_mask_fn

    def load_state(self, state: list[int]) -> None:
        """
        Loads the state of the environment. The state is a list of actions taken in the environment.

        The environment is reset and all actions in the state are performed in order to restore the environment to the
        same state.

        This works only for deterministic environments!

        :param state: the state to load
        :return: None
        """
        self.env.reset()
        self._wrapper_action_history = []

        for action in state:
            self.env.step(action)
            self._wrapper_action_history.append(action)

    def is_terminal(self) -> bool:
        """
        Returns True if the environment is in a terminal state, False otherwise.

        :return:
        """
        if not len(self.get_valid_actions()):
            return True
        else:
            return self._terminal_flag

    def action_masks(self) -> np.ndarray | None:
        """
        Returns the action masks for the environment. If the action_mask_fn is not set, it returns None.

        :return:
        """
        return self._action_mask_fn(self.env) if self._action_mask_fn is not None else None

    def get_valid_actions(self) -> list[int]:
        """
        Returns a list of valid actions for the current state of the environment.

        :return: a list of valid actions
        """
        if self._action_mask_fn is None:
            action_space: gym.spaces.Discrete = self.env.action_space  # Type hinting
            return list(range(action_space.n))
        else:
            return [i for i, mask in enumerate(self.action_masks()) if mask]

    def rollout(self) -> float:
        """
        Performs a random rollout from the current state of the environment and returns the return (sum of rewards)
        of the rollout.

        :return: the return of the rollout
        """
        log.debug("performing rollout")
        # random rollout
        # perform random valid action util terminal
        is_terminal_state = self.is_terminal()

        if is_terminal_state:
            _, _, _, _, info = self._step_tuple
            episode_return = info["episode"]["r"]
            return episode_return

        while not is_terminal_state:
            action = random.choice(self.get_valid_actions())
            # print(f"Valid actions: {self.get_valid_actions()}, selected action: {action}")
            _obs, _reward, is_terminal_state, _truncated, info = self.step(action)

        episode_return = info["episode"]["r"]
        log.debug(f"Rollout return: {episode_return}")
        return episode_return

    def get_state(self) -> list[int]:
        """
        Returns the current state of the environment. The state is a list of actions taken in the environment,
        namely all action that have been taken in the environment so far (since the last reset).

        :return: a list of actions taken in the environment
        """

        return self._wrapper_action_history.copy()

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a step in the environment. It adds the action to the action history and updates the terminal flag.

        :param action: action to perform in the environment
        :return: the step tuple of the environment (obs, reward, terminated, truncated, info)
        """
        step_tuple = self.env.step(action)
        self._wrapper_action_history.append(action)
        obs, reward, terminated, truncated, info = step_tuple

        self._terminal_flag = terminated or truncated
        self._step_tuple = step_tuple

        return step_tuple
