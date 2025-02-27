
from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_deterministic_wrapper import DeterministicSoloMCTSGymEnvWrapper
from gymcts.gymcts_gym_env import SoloMCTSGymEnv
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward
from gymcts.logger import log

import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    log.setLevel(20)


    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        reward_function="makespan",
    )
    # map reward to [1, -inf]
    # ideally you want the reward to be in the range of [-1, 1] for the UBC score
    env = TransformReward(env, lambda r: r / 55.0 + 2 if r != 0 else 0.0)
    env.reset()



    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.valid_action_mask()

    env = NaiveSoloMCTSGymEnvWrapper(
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
