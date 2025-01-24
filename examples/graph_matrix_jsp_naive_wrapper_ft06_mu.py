import sys

import gymnasium as gym
import numpy as np

from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.wrappers import TransformReward
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

if __name__ == '__main__':

    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        reward_function="makespan_scaled_by_lb",
        c_lb=ft06_makespan,
    )
    # reward should range from -1 to 1 in order to work with the ubc score
    # otherwise the c value should be adjusted
    # instead of adjusting c value of the ubc score, we can adjust the reward function
    env = TransformReward(env, lambda r: r + 2 if r != 0 else r)
    env.reset()

    #for i in range(37):
    #   obs , rew, term, trun, info = env.step(i)
    #   env.render()
    #   print(f"reward: {rew}")

    #sys.exit(1)


    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.unwrapped.valid_action_mask()


    env = NaiveSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )


    agent = SoloMCTSAgent(
        env=env,
        number_of_simulations_per_step=100,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=False,
        exclude_unvisited_nodes_from_render=True
    )

    log.setLevel(20)
    actions = agent.solve()

    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    print(f"makespan: {env.unwrapped.get_makespan()}")