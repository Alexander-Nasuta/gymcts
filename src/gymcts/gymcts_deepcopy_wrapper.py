import random
import copy

import numpy as np
from typing import TypeVar, Any, SupportsFloat, Callable
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from gymcts.gymcts_env_abc import GymctsABC

from gymcts.logger import log


class DeepCopyMCTSGymEnvWrapper(GymctsABC, gym.Wrapper):
    """
    A wrapper for gym environments that implements the GymctsABC interface.
    It uses deepcopys as state representation.
    Please note that this is not the most efficient way to implement the state representation.
    It is supposed to be used to see if your use-case works well with the MCTS algorithm.
    If it does, you can consider implementing all GymctsABC methods in a more efficient way.
    """

    # helper attributes for the wrapper
    _terminal_flag:bool = False
    _last_reward: SupportsFloat = 0
    _step_tuple: tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]] = None

    _action_mask_fn: Callable[[gym.Env], np.ndarray] | None = None

    def is_terminal(self) -> bool:
        """
        Returns True if the environment is in a terminal state, False otherwise.

        :return: True if the environment is in a terminal state, False otherwise.
        """
        return self._terminal_flag

    def load_state(self, state: Any) -> None:
        """
        The load_state method is not implemented. The state is loaded by replacing the env with the 'state' (the copy
        provided my 'get_state'). 'self' in a method cannot be replaced with another object (as far as i know).

        :param state: a deepcopy of the environment
        :return: None
        """
        msg = """
        The NaiveSoloMCTSGymEnvWrapper uses deepcopies of the entire env as the state.
        The loading of the state is done by replacing the env with the 'state' (the copy provided my 'get_state').
        'self' in a method cannot be replaced with another object (as far as i know). Therefore the copy is done by
        MCTSaAgent here.
        """
        raise NotImplementedError(msg)

    def __init__(self,
                 env,
                 action_mask_fn: str | Callable[[gym.Env], np.ndarray] | None = None,
                 buffer_length: int = 100,
                 record_video: bool = False,
                 ):
        """
        The constructor of the wrapper. It wraps the environment with RecordEpisodeStatistics and checks if the action
        space is discrete. It also checks if the action_mask_fn is a string or a callable. If it is a string, it tries to
        find the method in the environment. If it is a callable, it assigns it to the _action_mask_fn attribute.

        :param env: the environment to wrap
        :param action_mask_fn:
        :param buffer_length:
        :param record_video:
        """
        # wrap with RecordEpisodeStatistics if it is not already wrapped
        env = RecordEpisodeStatistics(env, buffer_length=buffer_length)

        gym.Wrapper.__init__(self, env)
        # super().__init__(env)

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

    def get_state(self) -> Any:
        """
        Returns the current state of the environment as a deepcopy of the environment.
        :return: a deepcopy of the environment
        """
        log.debug("getting state")
        original_state = self
        copied_state = copy.deepcopy(self)

        log.debug(f"original state memory location: {hex(id(original_state))}")
        log.debug(f"copied memory location: {hex(id(copied_state))}")

        return copied_state

    def action_masks(self) -> np.ndarray | None:
        """
        Returns the action masks for the environment. If the action_mask_fn is not set, it returns None.
        :return: the action masks for the environment
        """
        return self._action_mask_fn(self.env) if self._action_mask_fn is not None else None

    def get_valid_actions(self) -> list[int]:
        """
        Returns a list of valid actions for the current state of the environment.
        This used to obtain potential actions/subsequent sates for the MCTS tree.

        :return: the list of valid actions
        """
        if self._action_mask_fn is None:
            action_space: gym.spaces.Discrete = self.env.action_space  # Type hinting
            return list(range(action_space.n))
        else:
            return [i for i, mask in enumerate(self.action_masks()) if mask]

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a step in the environment.
        This method is used to update the wrapper with the new state and the new action, to realize the terminal state
        functionality.

        :param action: action to perform in the environment
        :return: the step tuple of the environment (obs, reward, terminated, truncated, info)
        """
        step_tuple = self.env.step(action)

        obs, reward, terminated, truncated, info = step_tuple
        self._terminal_flag = terminated or truncated
        self._step_tuple = step_tuple

        return step_tuple


    def rollout(self) -> float:
        """
        Performs a rollout from the current state of the environment and returns the return (sum of rewards) of the
        rollout.

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
