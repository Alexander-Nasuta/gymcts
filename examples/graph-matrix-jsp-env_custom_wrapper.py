from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward

from gymcts.gymcts_node import GymctsNode
from gymcts.logger import log

import gymnasium as gym
import numpy as np

class GraphMatrixJspGYMCTSWrapper(GymctsABC, gym.Wrapper):

    def __init__(self, env: DisjunctiveGraphJspEnv):
        super().__init__(env)

    def load_state(self, state: np.ndarray) -> None:
        self.env.unwrapped.load_state(state)

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal_state()

    def get_valid_actions(self) -> list[int]:
        return self.env.unwrapped.valid_action_list()

    def rollout(self) -> float:
        return self.env.unwrapped.greedy_rollout()

    def get_state(self) -> np.ndarray:
        return self.env.unwrapped.get_state()

    def action_masks(self) -> np.ndarray | None:
        """Return the action mask for the current state."""
        return self.env.unwrapped.valid_action_mask()


if __name__ == '__main__':
    log.setLevel(20)

    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        c_lb=ft06_makespan,
        reward_function="mcts", # this reward is in range [-inf, 1]
    )

    env.reset()

    env = GraphMatrixJspGYMCTSWrapper(
        env
    )

    import math
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
            f_x0_plus_z=1,
            z=36,  # number of steps to decay
        )
        res =max(2, int(res + 0.5))  # ensure at least 1 simulation
        return res

    GymctsNode.score_variate = "UCT_v0"

    agent = GymctsAgent(
        env=env,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        clear_mcts_tree_after_step=False,
        number_of_simulations_per_step=10,
        #calc_number_of_simulations_per_step=calc_simulations_per_step,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")
