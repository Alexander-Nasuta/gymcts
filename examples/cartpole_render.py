import gymnasium as gym

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

if __name__ == '__main__':
    # Create the environment and wrap it with RecordVideo
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", episode_trigger=lambda episode_id: True)

    env = NaiveSoloMCTSGymEnvWrapper(
        env,
    )

    # Reset the environment
    env.reset(seed=42)

    agent = SoloMCTSAgent(
        env=env,
        number_of_simulations_per_step=50,
        clear_mcts_tree_after_step=True,
    )

    terminal = False
    while not terminal:
        action, _ = agent.perform_mcts_step()
        obs, rew, term, trun, info = env.step(action)
        # obs, rew, term, trun, info = env.step(env.action_space.sample())
        terminal = term or trun

    env.close()
