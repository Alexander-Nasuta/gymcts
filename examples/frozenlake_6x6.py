import gymnasium as gym

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

from gymcts.logger import log

log.setLevel(20)

if __name__ == '__main__':
    log.debug("Starting example")

    # 0. create the environment
    custom_map = [
        "SFFFFF",
        "FFFFFF",
        "FFFHFF",
        "FFFFFH",
        "FHFFFF",
        "FFFFFG"
    ]
    env = gym.make(
        'FrozenLake-v1',
        desc=custom_map,
        map_name=None,
        is_slippery=True,
        render_mode="ansi"
    )
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = DeepCopyMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent = GymctsAgent(env=env, clear_mcts_tree_after_step=False)

    # 3. solve the environment
    actions = agent.solve(num_simulations_per_step=200)

    # 4. render the environment solution in the terminal
    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())
