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
        gym.Wrapper.__init__(self, env)

    def load_state(self, state: np.ndarray) -> None:
        self.env.unwrapped.load_state(state)

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal_state()

    def get_valid_actions(self) -> list[int]:
        return self.env.unwrapped.valid_action_mask()

    def rollout(self) -> float:
        return self.env.unwrapped.random_rollout()

    def get_state(self) -> np.ndarray:
        return env.unwrapped.get_state


if __name__ == '__main__':
    log.setLevel(20)

    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        reward_function="makespan",
    )
    # map reward to [1, -inf]
    # ideally you want the reward to be in the range of [-1, 1] for the UBC score
    env = TransformReward(env, lambda r: r / ft06_makespan + 2 if r != 0 else 0.0)
    env.reset()


    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.valid_action_mask()


    env = DeepCopyMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = GymctsAgent(
        env=env,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=1000,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")
