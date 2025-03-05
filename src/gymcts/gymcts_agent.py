import copy
import gymnasium as gym

from typing import TypeVar, Any, SupportsFloat, Callable

from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.gymcts_node import GymctsNode
from gymcts.gymcts_tree_plotter import _generate_mcts_tree

from gymcts.logger import log

TSoloMCTSNode = TypeVar("TSoloMCTSNode", bound="SoloMCTSNode")


class GymctsAgent:
    render_tree_after_step: bool = False
    render_tree_max_depth: int = 2
    exclude_unvisited_nodes_from_render: bool = False
    number_of_simulations_per_step: int = 25

    env: GymctsABC
    search_root_node: GymctsNode  # NOTE: this is not the same as the root of the tree!
    clear_mcts_tree_after_step: bool

    def __init__(self,
                 env: GymctsABC,
                 clear_mcts_tree_after_step: bool = True,
                 render_tree_after_step: bool = False,
                 render_tree_max_depth: int = 2,
                 number_of_simulations_per_step: int = 25,
                 exclude_unvisited_nodes_from_render: bool = False
                 ):
        # check if action space of env is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space must be discrete.")

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
            temp_node = max(temp_node.children.values(), key=lambda child: child.ucb_score())
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

        while not current_node.terminal:
            next_action, current_node = self.perform_mcts_step(num_simulations=num_simulations_per_step,
                                                               render_tree_after_step=render_tree_after_step)
            log.info(f"selected action {next_action} after {num_simulations_per_step} simulations.")
            action_list.append(next_action)
            log.info(f"current action list: {action_list}")

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


