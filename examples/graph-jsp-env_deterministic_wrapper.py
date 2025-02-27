from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_deterministic_wrapper import DeterministicSoloMCTSGymEnvWrapper
from gymnasium.core import WrapperActType, WrapperObsType
from gymcts.gymcts_gym_env import SoloMCTSGymEnv
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward, RecordEpisodeStatistics
from gymcts.logger import log

from typing import TypeVar, Any, SupportsFloat, Callable

import gymnasium as gym
import numpy as np

import random

if __name__ == '__main__':
    log.setLevel(20)

    env_kwargs = {
        "jps_instance": ft06,
        "default_visualisations": ["gantt_console", "graph_console"],
        "reward_function_parameters": {
            "scaling_divisor": 55.0
        }
    }

    env = DisjunctiveGraphJspEnv(**env_kwargs)
    # map reward to [1, -inf]
    # ideally you want the reward to be in the range of [-1, 1] for the UBC score
    env = TransformReward(env, lambda r: r / 55.0 + 2 if r is not 0 else 0.0)
    env.reset()


    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.valid_action_mask()


    env = DeterministicSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = SoloMCTSAgent(
        env=env,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=125,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")
