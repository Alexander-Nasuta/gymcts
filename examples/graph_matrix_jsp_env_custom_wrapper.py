from copy import copy
from typing import Any

import gymnasium as gym
import numpy as np

from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_gym_env import SoloMCTSGymEnv
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log


class GraphMatrixGYMCTSWrapper(SoloMCTSGymEnv, gym.Wrapper):

    def __init__(self, env: DisjunctiveGraphJspEnv):
        gym.Wrapper.__init__(self, env)
        env.valid_action_list()

    def load_state(self, state: Any) -> None:
        self.env.unwrapped.load_state(copy(state))

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal_state()

    def get_valid_actions(self) -> list[int]:
        return self.env.unwrapped.valid_action_list()

    def rollout(self) -> float:
        self.env.unwrapped.random_rollout()
        return self.env.unwrapped.get_makespan() / -55 + 2

    def get_state(self) -> Any:
        return env.unwrapped.get_state()


if __name__ == '__main__':
    log.setLevel(20)

    two_job_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]
    ], dtype=np.int32)

    env = DisjunctiveGraphJspEnv(
        #jsp_instance=two_job_jsp_instance,
        jsp_instance=ft06,
        reward_function="makespan_scaled_by_lb",
        c_lb=ft06_makespan,
        #c_lb=40,
    )
    env.reset()

    env = GraphMatrixGYMCTSWrapper(env)

    agent = SoloMCTSAgent(
        env=env,
        number_of_simulations_per_step=100,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")


