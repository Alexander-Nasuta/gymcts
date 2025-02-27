import random
import copy

import numpy as np
from typing import TypeVar, Any, SupportsFloat, Callable
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.wrappers import RecordEpisodeStatistics

from gymcts.gymcts_gym_env import SoloMCTSGymEnv

from gymcts.logger import log


class NaiveSoloMCTSGymEnvWrapper(SoloMCTSGymEnv, gym.Wrapper):


    _terminal_flag:bool = False
    _last_reward: SupportsFloat = 0
    _step_tuple: tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]] = None

    _action_mask_fn: Callable[[gym.Env], np.ndarray] | None = None

    def is_terminal(self) -> bool:
        return self._terminal_flag

    def load_state(self, state: Any) -> None:
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
        log.debug("getting state")
        original_state = self
        copied_state = copy.deepcopy(self)

        log.debug(f"original state memory location: {hex(id(original_state))}")
        log.debug(f"copied memory location: {hex(id(copied_state))}")

        return copied_state

    def action_masks(self) -> np.ndarray | None:
        return self._action_mask_fn(self.env) if self._action_mask_fn is not None else None

    def get_valid_actions(self) -> list[int]:
        if self._action_mask_fn is None:
            action_space: gym.spaces.Discrete = self.env.action_space  # Type hinting
            return list(range(action_space.n))
        else:
            return [i for i, mask in enumerate(self.action_masks()) if mask]

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        step_tuple = self.env.step(action)

        obs, reward, terminated, truncated, info = step_tuple
        self._terminal_flag = terminated or truncated
        self._step_tuple = step_tuple

        return step_tuple


    def rollout(self) -> float:
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
