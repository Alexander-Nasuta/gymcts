from typing import SupportsFloat, Any

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.core import WrapperActType, WrapperObsType
from jsp_instance_utils.instances import ft06, ft06_makespan
from stable_baselines3.common.callbacks import BaseCallback

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward

from gymcts.gymcts_neural_agent import GymctsNeuralAgent
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

        log.info({
            "performed_timesteps": ExperimentWrapper.performed_timesteps,
            "makespan": self.env.unwrapped.get_makespan(),
            "visited_states": len(ExperimentNmctsAgent.visited_states_set),
        })

        wb.log({
            "performed_timesteps": ExperimentWrapper.performed_timesteps,
            "makespan": self.env.unwrapped.get_makespan(),
            "visited_states": len(ExperimentNmctsAgent.visited_states_set),
        })

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

        if term or trun:
            wb.log({
                "performed_timesteps": ExperimentWrapper.performed_timesteps,
                "makespan": self.env.unwrapped.get_makespan(),
                "visited_states": len(ExperimentNmctsAgent.visited_states_set),
            })

        return obs, rew, term, trun, info


class ExperimentNmctsAgent(GymctsNeuralAgent):
    """
    A GymCTS agent that uses the ExperimentWrapper to track the visited states and performed timesteps.
    """

    visited_states_set = set()
    visited_states_set_with_rollout = set()

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
            if node not in ExperimentNmctsAgent.visited_states_set:
                ExperimentNmctsAgent.visited_states_set.add(tuple(node.state.unwrapped.get_state().ravel()))

        # Log the number of performed timesteps
        log.info(f"Performed timesteps: {ExperimentWrapper.performed_timesteps}, "
                 f"states visited: {len(ExperimentNmctsAgent.visited_states_set)}")

        # Log the action to Weights & Biases
        wb.log({
            "performed_timesteps": ExperimentWrapper.performed_timesteps,
            "visited_states": len(ExperimentNmctsAgent.visited_states_set)
        })


        return action




class ExperimentCallback(BaseCallback):

    def __init__(self, *args, **kwargs) -> None:
        super(ExperimentCallback, self).__init__(*args, **kwargs)

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        last_obs = self.locals["new_obs"][0] #only one env in this case
        ExperimentNmctsAgent.visited_states_set.add(tuple(last_obs.ravel()))

        done = self.locals["dones"][0]  # only one env in this case
        if done:
            makespan = self.locals["infos"][0]["makespan"]
            wb.log({
                "visited_states": len(ExperimentNmctsAgent.visited_states_set),
                "makespan": makespan,
                "performed_timesteps": ExperimentWrapper.performed_timesteps,
            })
        return True




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

    import torch

    model_kwargs = {
        "gamma": 0.99013,
        "gae_lambda": 0.9,
        "normalize_advantage": True,
        "n_epochs": 28,
        "n_steps": 432,
        "max_grad_norm": 0.5,
        "learning_rate": 6e-4,
        "policy_kwargs": {
            "net_arch": {
                "pi": [90, 90],
                "vf": [90, 90],
            },
            "ortho_init": True,
            "activation_fn": torch.nn.ELU,
            "optimizer_kwargs": {
                "eps": 1e-7
            }
        }
    }

    agent_kwargs = {
        "clear_mcts_tree_after_step": True,
        "render_tree_after_step": True,
        "render_tree_max_depth": 2,
        "number_of_simulations_per_step": 100,
        "exclude_unvisited_nodes_from_render": False,
        "calc_number_of_simulations_per_step": None,
        "score_variate": "PUCT_v0",
        "best_action_weight": 0.05,
        "model_kwargs": model_kwargs,
    }

    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload videos, imgs, files etc.
            save_code=False,  # optional
            project="test",  # specify your project here
    ) as run:


        env = DisjunctiveGraphJspEnv(**env_kwargs)
        print(f"env shape: {env.observation_space.shape}")
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

        agent = ExperimentNmctsAgent(
            env=env,
            **agent_kwargs
        )

        cb = ExperimentCallback()
        agent.learn(
            total_timesteps=10_000,
            callback=cb,
        )

        root = agent.search_root_node.get_root()

        actions = agent.solve(render_tree_after_step=True)
        for a in actions:
            obs, rew, term, trun, info = env.step(a)

        env.render()
        makespan = env.unwrapped.get_makespan()
        print(env.unwrapped.network_as_dataframe().to_dict(orient='records'))
        print(f"makespan: {makespan}")


