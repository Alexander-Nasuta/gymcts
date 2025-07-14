import copy
import random
import gymnasium as gym

from typing import TypeVar, Any, SupportsFloat, Callable, Literal

from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.gymcts_node import GymctsNode
from gymcts.gymcts_tree_plotter import _generate_mcts_tree

from gymcts.logger import log






class GymctsAgent:
    render_tree_after_step: bool = False
    render_tree_max_depth: int = 2
    exclude_unvisited_nodes_from_render: bool = False
    number_of_simulations_per_step: int = 25

    env: GymctsABC
    search_root_node: GymctsNode  # NOTE: this is not the same as the root of the tree!
    clear_mcts_tree_after_step: bool


    # (num_simulations: int, step_idx: int) -> int
    @staticmethod
    def calc_number_of_simulations_per_step(num_simulations: int, step_idx: int) -> int:
        """
        A function that returns a constant number of simulations per step.

        :param num_simulations: The number of simulations to return.
        :param step_idx: The current step index (not used in this function).
        :return: A callable that takes an environment as input and returns the constant number of simulations.
        """
        return num_simulations

    def __init__(self,
                 env: GymctsABC,
                 clear_mcts_tree_after_step: bool = True,
                 render_tree_after_step: bool = False,
                 render_tree_max_depth: int = 2,
                 number_of_simulations_per_step: int = 25,
                 exclude_unvisited_nodes_from_render: bool = False,
                 calc_number_of_simulations_per_step: Callable[[int,int], int] = None,
                 score_variate: Literal["UCT_v0", "UCT_v1", "UCT_v2",] = "UCT_v0",
                 best_action_weight=None,
                 ):
        # check if action space of env is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space must be discrete.")
        if calc_number_of_simulations_per_step is not None:
            # check if the provided function is callable
            if not callable(calc_number_of_simulations_per_step):
                raise ValueError("calc_number_of_simulations_per_step must be a callable accepting two arguments: num_simulations and step_idx.")
            # assign the provided function to the attribute
            # it needs to be staticmethod to be used as a class attribute
            print("Using provided calc_number_of_simulations_per_step function.")
            self.calc_number_of_simulations_per_step = staticmethod(calc_number_of_simulations_per_step)
        if score_variate not in ["UCT_v0", "UCT_v1", "UCT_v2"]:
            raise ValueError("score_variate must be one of ['UCT_v0', 'UCT_v1', 'UCT_v2'].")
        GymctsNode.score_variate = score_variate

        if best_action_weight is not None:
            if best_action_weight < 0 or best_action_weight > 1:
                raise ValueError("best_action_weight must be in range [0, 1].")
            GymctsNode.best_action_weight = best_action_weight


        self.render_tree_after_step = render_tree_after_step
        self.exclude_unvisited_nodes_from_render = exclude_unvisited_nodes_from_render
        self.render_tree_max_depth = render_tree_max_depth

        self.number_of_simulations_per_step = number_of_simulations_per_step

        self.env = env
        self.clear_mcts_tree_after_step = clear_mcts_tree_after_step

        self.search_root_node = GymctsNode(
            action=None,
            parent=None,
            env_reference=env,
        )

    def navigate_to_leaf(self, from_node: GymctsNode) -> GymctsNode:
        log.debug(f"Navigate to leaf. from_node: {from_node}")
        if from_node.terminal:
            log.debug("Node is terminal. Returning from_node")
            return from_node
        if from_node.is_leaf():
            log.debug("Node is leaf. Returning from_node")
            return from_node

        temp_node = from_node
        # NAVIGATION STRATEGY
        # select child with highest UCB score
        while not temp_node.is_leaf():
            children = list(temp_node.children.values())
            max_ucb_score = max(child.tree_policy_score() for child in children)
            best_children = [child for child in children if child.tree_policy_score() == max_ucb_score]
            temp_node = random.choice(best_children)
        log.debug(f"Selected leaf node: {temp_node}")
        return temp_node

    def expand_node(self, node: GymctsNode) -> None:
        log.debug(f"expanding node: {node}")
        # EXPANSION STRATEGY
        # expand all children

        child_dict = {}
        for action in node.valid_actions:
            # reconstruct state
            # load state of leaf node
            self._load_state(node)

            obs, reward, terminal, truncated, _ = self.env.step(action)
            child_dict[action] = GymctsNode(
                action=action,
                parent=node,
                env_reference=self.env,
            )
        node.children = child_dict

    def solve(self, num_simulations_per_step: int = None, render_tree_after_step: bool = None) -> list[int]:

        if num_simulations_per_step is None:
            num_simulations_per_step = self.number_of_simulations_per_step
        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        log.debug(f"Solving from root node: {self.search_root_node}")

        current_node = self.search_root_node

        action_list = []

        idx = 0
        while not current_node.terminal:
            num_sims = self.calc_number_of_simulations_per_step(num_simulations_per_step, idx)

            log.info(f"Performing MCTS step {idx} with {num_sims} simulations.")

            next_action, current_node = self.perform_mcts_step(num_simulations=num_sims,
                                                               render_tree_after_step=render_tree_after_step)
            log.info(f"selected action {next_action} after {num_sims} simulations.")
            action_list.append(next_action)
            log.info(f"current action list: {action_list}")

            idx += 1

        log.info(f"Final action list: {action_list}")
        # restore state of current node
        return action_list

    def _load_state(self, node: GymctsNode) -> None:
        if isinstance(self.env, DeepCopyMCTSGymEnvWrapper):
            self.env = copy.deepcopy(node.state)
        else:
            self.env.load_state(node.state)

    def perform_mcts_step(self, search_start_node: GymctsNode = None, num_simulations: int = None,
                          render_tree_after_step: bool = None) -> tuple[int, GymctsNode]:

        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        if num_simulations is None:
            num_simulations = self.number_of_simulations_per_step

        if search_start_node is None:
            search_start_node = self.search_root_node

        action = self.vanilla_mcts_search(
            search_start_node=search_start_node,
            num_simulations=num_simulations,
        )
        next_node = search_start_node.children[action]

        if self.clear_mcts_tree_after_step:
            # to clear memory we need to remove all nodes except the current node
            # this is done by setting the root node to the current node
            # and setting the parent of the current node to None
            # we also need to reset the children of the current node
            # this is done by calling the reset method
            next_node.reset()
        else:
            next_node.remove_parent()

        self.search_root_node = next_node

        return action, next_node

    def vanilla_mcts_search(self, search_start_node: GymctsNode = None, num_simulations=10) -> int:
        log.debug(f"performing one MCTS search step with {num_simulations} simulations")
        if search_start_node is None:
            search_start_node = self.search_root_node

        for i in range(num_simulations):
            log.debug(f"simulation {i}")
            # navigate to leaf
            leaf_node = self.navigate_to_leaf(from_node=search_start_node)

            if leaf_node.visit_count > 0 and not leaf_node.terminal:
                # expand leaf
                self.expand_node(leaf_node)
                leaf_node = leaf_node.get_random_child()

            # load state of leaf node
            self._load_state(leaf_node)

            # rollout
            episode_return = self.env.rollout()
            # self.env.render()

            self.backpropagation(node=leaf_node, episode_return=episode_return)

        if self.render_tree_after_step:
            self.show_mcts_tree()

        return search_start_node.get_best_action()

    def show_mcts_tree(self, start_node: GymctsNode = None, tree_max_depth: int = None) -> None:

        if start_node is None:
            start_node = self.search_root_node

        if tree_max_depth is None:
            tree_max_depth = self.render_tree_max_depth

        print(start_node.__str__(colored=True, action_space_n=self.env.action_space.n))
        for line in _generate_mcts_tree(
                start_node=start_node,
                depth=tree_max_depth,
                action_space_n=self.env.action_space.n,
        ):
            print(line)

    def show_mcts_tree_from_root(self, tree_max_depth: int = None) -> None:
        self.show_mcts_tree(start_node=self.search_root_node.get_root(), tree_max_depth=tree_max_depth)

    def backpropagation(self, node: GymctsNode, episode_return: float) -> None:
        log.debug(f"performing backpropagation from leaf node: {node}")
        while not node.is_root():
            # node.mean_value = ((node.mean_value * node.visit_count) + episode_return) / (node.visit_count + 1)
            node.mean_value = node.mean_value + (episode_return - node.mean_value) / (node.visit_count + 1)
            node.visit_count += 1
            node.max_value = max(node.max_value, episode_return)
            node.min_value = min(node.min_value, episode_return)
            node = node.parent
        # also update root node
        # node.mean_value = ((node.mean_value * node.visit_count) + episode_return) / (node.visit_count + 1)
        node.mean_value = node.mean_value + (episode_return - node.mean_value) / (node.visit_count + 1)
        node.visit_count += 1
        node.max_value = max(node.max_value, episode_return)
        node.min_value = min(node.min_value, episode_return)


