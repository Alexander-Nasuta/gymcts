import gymnasium as gym
import numpy as np

from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

if __name__ == '__main__':
    log.debug("Starting example")

    env = DisjunctiveGraphJspEnv(jsp_instance=ft06)
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

    agent = SoloMCTSAgent(env=env)

    actions = agent.solve(num_simulations_per_step=10)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        env.render()
    print(info["episode"]["r"])

