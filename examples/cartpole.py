import gymnasium as gym

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

if __name__ == '__main__':

    # 0. create the environment
    env = gym.make("CartPole-v1")
    env.reset(seed=42)


    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = DeepCopyMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent = GymctsAgent(
        env=env,
        number_of_simulations_per_step=50,
        clear_mcts_tree_after_step=True,
    )

    # 3. solve the environment
    terminal = False
    step = 0
    while not terminal:
        action, _ = agent.perform_mcts_step()
        obs, rew, term, trun, info = env.step(action)
        terminal = term or trun

        step += 1

        # log to console every 10 steps
        if step % 10 == 0:
            print(f"reward: {rew}, step: {step}")

    if step >= 475:
        print("CartPole-v1 successfully balanced!")
    else:
        print("CartPole-v1 failed to balance.")



