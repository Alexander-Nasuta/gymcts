
import gymnasium as gym

from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

log.setLevel(20)





if __name__ == '__main__':
    log.debug("Starting example")

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
    env.reset()



    env = NaiveSoloMCTSGymEnvWrapper(env)

    agent = SoloMCTSAgent(env=env, clear_mcts_tree_after_step=False)

    print(env.render())
    actions = agent.solve(num_simulations_per_step=50)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())
