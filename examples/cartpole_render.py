import gymnasium as gym

if __name__ == '__main__':
    # Create the environment and wrap it with RecordVideo
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", episode_trigger=lambda episode_id: True)

    # Reset the environment
    env.reset(seed=42)

    terminal = False
    while not terminal:
        obs, rew, term, trun, info = env.step(env.action_space.sample())
        terminal = term or trun

    env.close()