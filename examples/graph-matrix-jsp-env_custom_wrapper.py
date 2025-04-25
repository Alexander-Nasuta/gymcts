from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward
from gymcts.logger import log

import gymnasium as gym
import numpy as np

class GraphMatrixJspGYMCTSWrapper(GymctsABC, gym.Wrapper):

    def __init__(self, env: DisjunctiveGraphJspEnv):
        super().__init__(env)

    def load_state(self, state: np.ndarray) -> None:
        self.env.unwrapped.load_state(state)

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal_state()

    def get_valid_actions(self) -> list[int]:
        return self.env.unwrapped.valid_action_list()

    def rollout(self) -> float:
        return self.env.unwrapped.greedy_rollout()

    def get_state(self) -> np.ndarray:
        return self.env.unwrapped.get_state()


if __name__ == '__main__':
    log.setLevel(20)

    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        c_lb=ft06_makespan,
        reward_function="mcts", # this reward is in range [-inf, 1]
    )

    env.reset()

    env = GraphMatrixJspGYMCTSWrapper(
        env
    )

    agent = GymctsAgent(
        env=env,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=25,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")
