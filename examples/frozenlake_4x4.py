
import gymnasium as gym


from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

log.setLevel(20)





if __name__ == '__main__':
    log.debug("Starting example")

    # 0. create the environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = NaiveSoloMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True
    )

    # 3. solve the environment
    actions = agent.solve()

    # 4. render the environment solution in the terminal
    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())

    # 5. print the solution
    # read the solution from the info provided by the RecordEpisodeStatistics wrapper (that NaiveSoloMCTSGymEnvWrapper wraps internally)
    episode_length = info["episode"]["l"]
    episode_return = info["episode"]["r"]

    if episode_return == 1.0:
        print(f"Environment solved in {episode_length} steps.")
    else:
        print(f"Environment not solved in {episode_length} steps.")
