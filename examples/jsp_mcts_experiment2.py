from typing import SupportsFloat, Any

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.core import WrapperActType, WrapperObsType
from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward

from gymcts.gymcts_node import GymctsNode
from gymcts.logger import log

import gymnasium as gym
import numpy as np

import random

import wandb as wb


class ExperimentWrapper(DeepCopyMCTSGymEnvWrapper, gym.Wrapper):
    performed_timesteps = 0

    def __init__(self, env: DisjunctiveGraphJspEnv, action_mask_fn=None):
        super().__init__(env=env,action_mask_fn=action_mask_fn)



    def rollout(self) -> float:
        if self.is_terminal():
            # If the environment is terminal, return the makespan as a reward
            lower_bound = self.env.unwrapped.reward_function_parameters['scaling_divisor']

            return -self.env.unwrapped.get_makespan() / lower_bound + 2

        number_of_steps_till_this_state = len(self.env.unwrapped.action_history)
        rest_of_episode_return, _ = self.env.unwrapped.random_rollout()

        log.debug(
            f"Performed timesteps: {ExperimentWrapper.performed_timesteps}, "
            f"Makespan: {self.env.unwrapped.get_makespan()}"
        )

        number_of_steps_till_terminal_state = len(self.env.unwrapped.action_history)

        num_steps_done_during_rollout = number_of_steps_till_terminal_state - number_of_steps_till_this_state
        ExperimentWrapper.performed_timesteps += num_steps_done_during_rollout


        return rest_of_episode_return + 2

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # call the parent step method
        obs, rew, term, trun, info = super().step(action)

        # increment the number of performed timesteps
        ExperimentWrapper.performed_timesteps += 1


        return obs, rew, term, trun, info


class ExperimentMctsAgent(GymctsAgent):
    """
    A GymCTS agent that uses the ExperimentWrapper to track the visited states and performed timesteps.
    """

    visited_states_set = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def vanilla_mcts_search(self, search_start_node: GymctsNode = None, num_simulations=10) -> int:
        """
        Perform a vanilla MCTS search with the given parameters.
        """
        # Call the parent method to perform the search
        action = super().vanilla_mcts_search(
            search_start_node=search_start_node,
            num_simulations=num_simulations
        )

        # Add the visited states to the set
        for node in search_start_node.traverse_nodes():
            if node not in ExperimentMctsAgent.visited_states_set:
                ExperimentMctsAgent.visited_states_set.add(tuple(node.state.unwrapped.get_state().ravel()))

        # Log the number of performed timesteps
        log.info(f"Performed timesteps: {ExperimentWrapper.performed_timesteps}, "
                 f"states visited: {len(ExperimentMctsAgent.visited_states_set)}")

        return action





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

    agent_kwargs = {
        "clear_mcts_tree_after_step": True,
        "render_tree_after_step": True,
        "render_tree_max_depth": 2,
        "number_of_simulations_per_step": 200,
        "exclude_unvisited_nodes_from_render": False,
        "calc_number_of_simulations_per_step": None,
        "score_variate": "UCT_v2",
        "best_action_weight": 0.05,
    }


    env = DisjunctiveGraphJspEnv(**env_kwargs)
    # map reward to [1, -inf]
    # ideally you want the reward to be in the range of [-1, 1] for the UBC score
    # env = TransformReward(env, lambda r: r / ft06_makespan + 2 if r != 0 else 0.0)
    env.reset()


    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.valid_action_mask()

    env = ExperimentWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = ExperimentMctsAgent(
        env=env,
        **agent_kwargs
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(env.unwrapped.network_as_dataframe().to_dict(orient='records'))
    print(f"makespan: {makespan}")


