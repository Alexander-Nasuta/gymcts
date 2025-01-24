from typing import TypeVar, Any, SupportsFloat, Callable
from abc import ABC, abstractmethod
import gymnasium as gym

TSoloMCTSNode = TypeVar("TSoloMCTSNode", bound="SoloMCTSNode")


class SoloMCTSGymEnv(ABC, gym.Env):

    @abstractmethod
    def get_state(self) -> Any:
        pass

    @abstractmethod
    def load_state(self, state: Any) -> None:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_valid_actions(self) -> list[int]:
        pass

    @abstractmethod
    def rollout(self) -> float:
        pass
