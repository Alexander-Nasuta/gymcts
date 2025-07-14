import copy
import sys
from typing import Any, Literal

import random
import math
import sb3_contrib

import gymnasium as gym
import numpy as np

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_instance_utils.instances import ft06, ft06_makespan
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_node import GymctsNode

from gymcts.logger import log


class GraphJspNeuralGYMCTSWrapper(GymctsABC, gym.Wrapper):

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


    def action_masks(self) -> np.ndarray | None:
        """Return the action mask for the current state."""
        return self.env.unwrapped.valid_action_mask()




class GymctsNeuralNode(GymctsNode):
    PUCT_v3_mu = 0.95

    MuZero_c1 = 1.25
    MuZero_c2 = 19652.0

    """
    PUCT (Predictor + UCT) exploration terms:

    PUCT_v0:
        c * P(s, a) * √( N(s) / (1 + N(s,a)) )

    PUCT_v1:
        c * P(s, a) * √( 2 * ln(N(s)) / N(s,a) )

    PUCT_v2:
        c * P(s, a) * √( N(s) ) / N(s,a)

    PUCT_v3:
        c * P(s, a)^μ * √( N(s) / (1 + N(s,a)) )

    PUCT_v4:
        c * ( P(s, a) / (1 + N(s,a)) )

    PUCT_v5:
        c * P(s, a) * ( √(N(s)) + 1 ) / (N(s,a) + 1)

    PUCT_v6:
        c * P(s, a) * N(s) / (1 + N(s,a))

    PUCT_v7:
        c * P(s, a) * ( √(N(s)) + ε ) / (N(s,a) + 1)

    PUCT_v8:
        c * P(s, a) * √( (ln(N(s)) + 1) / (1 + N(s,a)) )

    PUCT_v9:
        c * P(s, a) * √( N(s) / (1 + N(s,a)) )

    PUCT_v10:
        c * P(s, a) * √( ln(N(s)) / (1 + N(s,a)) )


    MuZero exploration terms:

    MuZero_v0:
        P(s, a) * √( N(s) / (1 + N(s,a)) ) * [ c₁ + ln( (N(s) + c₂ + 1) / c₂ ) ]

    MuZero_v1:
        P(s, a) * √( N(s) / (1 + N(s,a)) ) * [ c₁ + ln( (N(s) + c₂ + 1) / c₂ ) ]


    Where:
        - N(s):      number of times state s has been visited
        - N(s,a):    number of times action a was taken from state s
        - P(s,a):    prior probability of selecting action a from state s
        - c, c₁, c₂: exploration constants
        - μ:         exponent applied to P(s,a) in some variants
        - ε:         small constant to avoid division by zero (in PUCT 7)
    """
    score_variate: Literal[
        "PUCT_v0",
        "PUCT_v1",
        "PUTC_v2",
        "PUTC_v3",
        "PUTC_v4",
        "PUTC_v5",
        "PUTC_v6",
        "PUTC_v7",
        "PUTC_v8",
        "PUTC_v9",
        "PUTC_v10",
        "MuZero_v0",
        "MuZero_v1",
    ] = "PUCT_v0"

    def __init__(
            self,
            action: int,
            parent: 'GymctsNeuralNode',
            env_reference: GymctsABC,
            prior_selection_score: float,
            observation: np.ndarray | None = None,
        ):
        super().__init__(action, parent, env_reference)

        self._obs = observation
        self._selection_score_prior = prior_selection_score


    def tree_policy_score(self) -> float:
        # call the superclass (GymctsNode) for ucb_score
        c = GymctsNode.ubc_c
        # the way alpha zero does it
        # exploration_term = self._selection_score_prior * c * math.sqrt(math.log(self.parent.visit_count)) / (1 + self.visit_count)
        # the way the vanilla gymcts does it
        p_sa = self._selection_score_prior
        n_s = self.parent.visit_count
        n_sa = self.visit_count
        if GymctsNeuralNode.score_variate == "PUCT_v0":
            return self.mean_value + c * p_sa * math.sqrt(n_s) / (1 + n_sa)
        elif GymctsNeuralNode.score_variate == "PUCT_v1":
            return self.mean_value + c * p_sa * math.sqrt(2 * math.log(n_s) / (n_sa))
        elif GymctsNeuralNode.score_variate == "PUCT_v2":
            return self.mean_value + c * p_sa * math.sqrt(n_s) / n_sa
        elif GymctsNeuralNode.score_variate == "PUCT_v3":
            return self.mean_value + c * (p_sa ** GymctsNeuralNode.PUCT_v3_mu) * math.sqrt(n_s / (1 + n_sa))
        elif GymctsNeuralNode.score_variate == "PUCT_v4":
            return self.mean_value + c * (p_sa / (1 + n_sa))
        elif GymctsNeuralNode.score_variate == "PUCT_v5":
            return self.mean_value + c * p_sa * (math.sqrt(n_s) + 1) / (n_sa + 1)
        elif GymctsNeuralNode.score_variate == "PUCT_v6":
            return self.mean_value + c * p_sa * n_s / (1 + n_sa)
        elif GymctsNeuralNode.score_variate == "PUCT_v7":
            epsilon = 1e-8
            return self.mean_value + c * p_sa * (math.sqrt(n_s) + epsilon) / (n_sa + 1)
        elif GymctsNeuralNode.score_variate == "PUCT_v8":
            return self.mean_value + c * p_sa * math.sqrt((math.log(n_s) + 1) / (1 + n_sa))
        elif GymctsNeuralNode.score_variate == "PUCT_v9":
            return self.mean_value + c * p_sa * math.sqrt(n_s / (1 + n_sa))
        elif GymctsNeuralNode.score_variate == "PUCT_v10":
            return self.mean_value + c * p_sa * math.sqrt(math.log(n_s) / (1 + n_sa))
        elif GymctsNeuralNode.score_variate == "MuZero_v0":
            c1 = GymctsNeuralNode.MuZero_c1
            c2 = GymctsNeuralNode.MuZero_c2
            return self.mean_value + c * p_sa * math.sqrt(n_s) / (1 + n_sa) * (c1 + math.log((n_s + c2 + 1) / c2))
        elif GymctsNeuralNode.score_variate == "MuZero_v1":
            c1 = GymctsNeuralNode.MuZero_c1
            c2 = GymctsNeuralNode.MuZero_c2
            return self.mean_value + c * p_sa * math.sqrt(n_s) / (1 + n_sa) * (c1 + math.log((n_s + c2 + 1) / c2))


        exploration_term = self._selection_score_prior * c * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count)) if self.visit_count > 0 else float("inf")
        return self.mean_value + exploration_term


    def get_best_action(self) -> int:
        """
        Returns the best action of the node. The best action is the action with the highest score.
        The best action is the action that has the highest score.

        :return: the best action of the node.
        """
        return max(self.children.values(), key=lambda child: child.max_value).action


    def __str__(self, colored=False, action_space_n=None) -> str:
        """
        Returns a string representation of the node. The string representation is used for visualisation purposes.
        It is used for example in the mcts tree visualisation functionality.

        :param colored: true if the string representation should be colored, false otherwise. (ture is used by the mcts tree visualisation)
        :param action_space_n: the number of actions in the action space. This is used for coloring the action in the string representation.
        :return: a potentially colored string representation of the node.
        """
        if not colored:

            if not self.is_root():
                return f"(a={self.action}, N={self.visit_count}, Q_v={self.mean_value:.2f}, best={self.max_value:.2f}, ubc={self.tree_policy_score():.2f})"
            else:
                return f"(N={self.visit_count}, Q_v={self.mean_value:.2f}, best={self.max_value:.2f}) [root]"

        import gymcts.colorful_console_utils as ccu

        if self.is_root():
            return f"({ccu.CYELLOW}N{ccu.CEND}={self.visit_count}, {ccu.CYELLOW}Q_v{ccu.CEND}={self.mean_value:.2f}, {ccu.CYELLOW}best{ccu.CEND}={self.max_value:.2f})"

        if action_space_n is None:
            raise ValueError("action_space_n must be provided if colored is True")

        p = ccu.CYELLOW
        e = ccu.CEND
        v = ccu.CCYAN

        def colorful_value(value: float | int | None) -> str:
            if value == None:
                return f"{ccu.CGREY}None{e}"
            color = ccu.CCYAN
            if value == 0:
                color = ccu.CRED
            if value == float("inf"):
                color = ccu.CGREY
            if value == -float("inf"):
                color = ccu.CGREY

            if isinstance(value, float):
                return f"{color}{value:.2f}{e}"

            if isinstance(value, int):
                return f"{color}{value}{e}"

        root_node = self.get_root()
        mean_val = f"{self.mean_value:.2f}"


        return ((f"("
                 f"{p}a{e}={ccu.wrap_evenly_spaced_color(s=self.action, n_of_item=self.action, n_classes=action_space_n)}, "
                 f"{p}N{e}={colorful_value(self.visit_count)}, "
                 f"{p}Q_v{e}={ccu.wrap_with_color_scale(s=mean_val, value=self.mean_value, min_val=root_node.min_value, max_val=root_node.max_value)}, "
                 f"{p}best{e}={colorful_value(self.max_value)}") +
                (f", {p}{GymctsNeuralNode.score_variate}{e}={colorful_value(self.tree_policy_score())})" if not self.is_root() else ")"))



class GymctsNeuralAgent(GymctsAgent):

    def __init__(self,
                 env: GymctsABC,
                 *args,
                 model_kwargs=None,
                 score_variate: Literal[
                     "PUCT_v0",
                     "PUCT_v1",
                     "PUTC_v2",
                     "PUTC_v3",
                     "PUTC_v4",
                     "PUTC_v5",
                     "PUTC_v6",
                     "PUTC_v7",
                     "PUTC_v8",
                     "PUTC_v9",
                     "PUTC_v10",
                     "MuZero_v0",
                     "MuZero_v1",
                 ] = "PUCT_v0",
                 **kwargs
                 ):

        # init super class
        super().__init__(
            env=env,
            *args,
            **kwargs
        )
        if score_variate not in [
            "PUCT_v0", "PUCT_v1", "PUTC_v2",
            "PUTC_v3", "PUTC_v4", "PUTC_v5",
            "PUTC_v6", "PUTC_v7", "PUTC_v8",
            "PUTC_v9", "PUTC_v10",
            "MuZero_v0", "MuZero_v1"
        ]:
            raise ValueError(f"Invalid score_variate: {score_variate}. Must be one of: "
                             f"PUCT_v0, PUCT_v1, PUTC_v2, PUTC_v3, PUTC_v4, PUTC_v5, "
                             f"PUTC_v6, PUTC_v7, PUTC_v8, PUTC_v9, PUTC_v10, MuZero_v0, MuZero_v1")
        GymctsNeuralNode.score_variate = score_variate

        if model_kwargs is None:
            model_kwargs = {}
        obs, info = env.reset()

        self.search_root_node = GymctsNeuralNode(
            action=None,
            parent=None,
            env_reference=env,
            observation=obs,
            prior_selection_score=1.0,
        )

        def mask_fn(env: gym.Env) -> np.ndarray:
            mask = env.action_masks()
            if mask is None:
                mask = np.ones(env.action_space.n, dtype=np.float32)
            return mask

        env = ActionMasker(env, action_mask_fn=mask_fn)

        model_kwargs = {
            "policy": MaskableActorCriticPolicy,
            "env": env,
            "verbose": 1,
        } | model_kwargs

        self._model = sb3_contrib.MaskablePPO(**model_kwargs)





    def learn(self, total_timesteps:int, **kwargs) -> None:
        """Learn from the environment using the MaskablePPO model."""
        self._model.learn(total_timesteps=total_timesteps, **kwargs)


    def expand_node(self, node: GymctsNeuralNode) -> None:
        log.debug(f"expanding node: {node}")
        # EXPANSION STRATEGY
        # expand all children

        child_dict = {}

        self._load_state(node)

        obs_tensor, vectorized_env = self._model.policy.obs_to_tensor(np.array([node._obs]))
        action_masks = np.array([self.env.action_masks()])
        distribution = self._model.policy.get_distribution(obs=obs_tensor, action_masks=action_masks)
        unwrapped_distribution = distribution.distribution.probs[0]

        # print(f'valid actions: {node.valid_actions}')
        # print(f'env mask: {self.env.action_masks()}')
        # print(f'env valid actions: {self.env.get_valid_actions()}')
        """
                for action in node.valid_actions:
            # reconstruct state
            # load state of leaf node
            self._load_state(node)

            obs, reward, terminal, truncated, _ = self.env.step(action)
            child_dict[action] = GymctsNeuralNode(
                action=action,
                parent=node,
                env_reference=self.env,
                observation=obs,
                prior_selection_score=1.0,
            )
        node.children = child_dict
        return
        """

        for action, prob in enumerate(unwrapped_distribution):
            self._load_state(node)

            log.debug(f"Probabily for action {action}: {prob}")

            if prob == 0.0:
                continue


            assert action in node.valid_actions, f"Action {action} is not in valid actions: {node.valid_actions}"

            obs, reward, terminal, truncated, _ = self.env.step(action)
            child_dict[action] = GymctsNeuralNode(
                action=action,
                parent=node,
                observation=copy.deepcopy(obs),
                env_reference=self.env,
                prior_selection_score=float(prob)
            )

        node.children = child_dict
        # print(f"Expanded node {node} with {len(node.children)} children.")





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

    env = GraphJspNeuralGYMCTSWrapper(env)

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

    agent = GymctsNeuralAgent(
        env=env,
        render_tree_after_step=True,
        render_tree_max_depth=3,
        exclude_unvisited_nodes_from_render=False,
        number_of_simulations_per_step=15,
        # clear_mcts_tree_after_step = False,
        model_kwargs=model_kwargs
    )

    agent.learn(total_timesteps=10_000)


    agent.solve()

    actions = agent.solve(render_tree_after_step=True)
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.render()
    makespan = env.unwrapped.get_makespan()
    print(f"makespan: {makespan}")






