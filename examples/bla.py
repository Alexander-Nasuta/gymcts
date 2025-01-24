import gymnasium as gym

from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

log.setLevel(20)

if __name__ == '__main__':
    import numpy as np

    jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]
    ], dtype=np.int32)

    env = DisjunctiveGraphJspEnv(jsp_instance=jsp_instance)


    def mask_fn(env: gym.Env) -> np.ndarray:

        return env.unwrapped.valid_action_mask()

    env = NaiveSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )


    agent = SoloMCTSAgent(env=env, clear_mcts_tree_after_step=True)

    agent.solve(num_simulations_per_step=10)
