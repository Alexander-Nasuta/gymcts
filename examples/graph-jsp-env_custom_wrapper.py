from typing import Any

import random

import gymnasium as gym

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_gym_env import SoloMCTSGymEnv

from gymcts.logger import log


class GraphJspGYMCTSWrapper(SoloMCTSGymEnv, gym.Wrapper):

    def __init__(self, env: DisjunctiveGraphJspEnv):
        gym.Wrapper.__init__(self, env)

    def load_state(self, state: Any) -> None:
        self.env.reset()
        for action in state:
            self.env.step(action)

    def is_terminal(self) -> bool:
        return self.env.unwrapped.is_terminal()

    def get_valid_actions(self) -> list[int]:
        return list(self.env.unwrapped.valid_actions())

    def rollout(self) -> float:
        terminal = env.is_terminal()

        if terminal:
            lower_bound = env.unwrapped.reward_function_parameters['scaling_divisor']
            return - env.unwrapped.get_makespan() / lower_bound + 2

        reward = 0
        while not terminal:
            action = random.choice(self.get_valid_actions())
            obs, reward, terminal, truncated, _ = env.step(action)

        return reward + 2

    def get_state(self) -> Any:
        return env.unwrapped.get_action_history()


if __name__ == '__main__':
    log.setLevel(20)

    env_kwargs = {
        "jps_instance": ft06,
        "default_visualisations": ["gantt_console", "graph_console"],
        "reward_function_parameters": {
            "scaling_divisor": ft06_makespan
        },
        "reward_function": "nasuta",
    }

    env = DisjunctiveGraphJspEnv(**env_kwargs)
    env.reset()

    env = GraphJspGYMCTSWrapper(env)

    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=True,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=50,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")
