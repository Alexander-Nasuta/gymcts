import pytest

import gymnasium as gym

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper


def test_frozenlake_4x4():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = DeepCopyMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True
    )

    # 3. solve the environment
    actions = agent.solve()

    # 4. render the environment solution in the terminal
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    episode_return = info["episode"]["r"]

    assert episode_return >= 1.0


@pytest.mark.skip(reason="Sometimes the test fails due to the random nature of the environment.")
def test_cartpole():
    env = gym.make("CartPole-v1")
    env.reset()

    env = DeepCopyMCTSGymEnvWrapper(env)

    agent = GymctsAgent(
        env=env,
        number_of_simulations_per_step=50,
        clear_mcts_tree_after_step=True,
    )

    terminal = False
    step = 0
    while not terminal:
        action, _ = agent.perform_mcts_step()
        obs, rew, term, trun, info = env.step(action)
        terminal = term or trun
        step += 1

    assert step >= 475


def test_mountain_car_continuous():
    env = gym.make("MountainCarContinuous-v0")
    env.reset()

    with pytest.raises(ValueError):
        # continuous action spaces are not supported
        env = DeepCopyMCTSGymEnvWrapper(env)

