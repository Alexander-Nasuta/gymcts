import gymnasium as gym
import math

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

from gymcts.logger import log

# set log level to 20 (INFO)
# set log level to 10 (DEBUG) to see more detailed information
log.setLevel(20)

if __name__ == '__main__':
    # 0. create the environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = DeepCopyMCTSGymEnvWrapper(env)

    def calc_simulations_per_step(num_simulations: int, step_idx: int) -> int:
        def exponential_decay(x, x0, f_x0, f_x0_plus_z, z):
            """
            Evaluates the exponential decay function at x.

            Parameters:
                x: The input value where you want to evaluate the function.
                x0: The starting point.
                f_x0: The function value at x0.
                f_x0_plus_z: The function value at x0 + z.
                z: The number of steps from x0 to where f(x0 + z) is known.

            Returns:
                The decayed value f(x).
            """
            # Calculate decay rate k using the two known values
            k = math.log(f_x0 / f_x0_plus_z) / z

            # Compute f(x) = f_x0 * exp(-k * (x - x0))
            return f_x0 * math.exp(-k * (x - x0))

        res = exponential_decay(
            x=step_idx,
            x0=0,  # starting point
            f_x0=100,
            f_x0_plus_z=10,
            z=5,  # number of steps to decay
        )
        res =max(1, int(res + 0.5))  # ensure at least 1 simulation
        return res

    # 2. create the agent
    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=False,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True,
        calc_number_of_simulations_per_step=calc_simulations_per_step,
    )

    # 3. solve the environment
    actions = agent.solve()

    # 4. render the environment solution in the terminal
    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())

    # 5. print the solution
    # read the solution from the info provided by the RecordEpisodeStatistics wrapper
    # (that NaiveSoloMCTSGymEnvWrapper uses internally)
    episode_length = info["episode"]["l"]
    episode_return = info["episode"]["r"]

    if episode_return == 1.0:
        print(f"Environment solved in {episode_length} steps.")
    else:
        print(f"Environment not solved in {episode_length} steps.")
