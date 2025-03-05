import copy
import gymnasium as gym

from typing import TypeVar, Any, SupportsFloat, Callable

from ray.types import ObjectRef

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_env_abc import GymctsABC
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.gymcts_node import GymctsNode

from gymcts.logger import log

import ray
import copy

TSoloMCTSNode = TypeVar("TSoloMCTSNode", bound="SoloMCTSNode")


@ray.remote
def mcts_lookahead(
        gymcts_start_node: GymctsNode,
        ray_env_ref: GymctsABC,
        num_simulations: int) -> GymctsNode:
    print(f"type of gymcts_start_node: {type(gymcts_start_node)}")
    print(f"type of ray_env_ref: {type(ray_env_ref)}")
    env_copy = copy.deepcopy(ray_env_ref)
    gymcts_start_node = copy.deepcopy(gymcts_start_node)

    agent = GymctsAgent(
        env=env_copy,
        clear_mcts_tree_after_step=False,
        number_of_simulations_per_step=num_simulations,
    )
    agent.search_root_node = gymcts_start_node

    agent.vanilla_mcts_search(
        search_start_node=gymcts_start_node,
        num_simulations=num_simulations,
    )
    return agent.search_root_node


@ray.remote(num_cpus=1)
def merge_trees(
        gymcts_tree_merge_node1: GymctsNode,
        gymcts_tree_merge_node2: GymctsNode) -> GymctsNode:
    resulting_tree_node = merge_nodes(
        gymcts_tree_merge_node1,
        gymcts_tree_merge_node2
    )
    # update gymcts_tree_merge_node1 with the resulting tree
    return resulting_tree_node


def merge_nodes(gymcts_node1, gymcts_node2, perform_state_equality_check=False):
    log.debug(f"merging {gymcts_node1} and {gymcts_node2}")
    # maybe add some state equality check here
    if perform_state_equality_check:
        if gymcts_node1.state != gymcts_node2.state:
            raise ValueError("States are different")

    if gymcts_node1 is None:
        log.debug(f"first node is None, returning second node ({gymcts_node2})")
        return gymcts_node2
    if gymcts_node2 is None:
        log.debug(f"second node is None, returning first node ({gymcts_node1})")
        return gymcts_node1
    if gymcts_node1 is None and gymcts_node2 is None:
        log.error("Both nodes are None")
        raise ValueError("Both nodes are None")

    if gymcts_node1.is_leaf() and not gymcts_node2.is_leaf():
        log.debug(f"first node is leaf, second node is not leaf")
        gymcts_node2.parent = gymcts_node1.parent
        log.debug(f"returning first node: {gymcts_node2}")
        return gymcts_node2

    if gymcts_node2.is_leaf() and not gymcts_node1.is_leaf():
        log.debug(f"second node is leaf, first node is not leaf")
        log.debug(f"returning first node: {gymcts_node1}")
        return gymcts_node1

    if gymcts_node1.is_leaf() and gymcts_node2.is_leaf():
        log.debug(f"both nodes are leafs, returning first node")
        log.debug(f"returning first node: {gymcts_node1}")
        return gymcts_node1

    # check if gymcts_node1 and gymcts_node2 have the same children
    if gymcts_node1.children.keys() != gymcts_node2.children.keys():
        log.error("Nodes have different children")
        raise ValueError("Nodes have different children")

    for (action1, child1), (action2, child2) in zip(gymcts_node1.children.items(), gymcts_node2.children.items()):
        if action1 != action2:
            log.error("Actions are different")
            raise ValueError("Actions are different")
        log.debug(f"merging children with action {action1} for node {gymcts_node1}")
        gymcts_node1.children[action1] = merge_nodes(
            child1,
            child2,
            perform_state_equality_check=perform_state_equality_check
        )

    visit_count = gymcts_node1.visit_count + gymcts_node2.visit_count
    mean_value = (
                         gymcts_node1.mean_value * gymcts_node1.visit_count + gymcts_node2.mean_value * gymcts_node2.visit_count) / visit_count
    max_value = max(gymcts_node1.max_value, gymcts_node2.max_value)
    min_value = min(gymcts_node1.min_value, gymcts_node2.min_value)

    gymcts_node1.visit_count = visit_count
    gymcts_node1.mean_value = mean_value
    gymcts_node1.max_value = max_value
    gymcts_node1.min_value = min_value
    log.debug(f"merged node: {gymcts_node1}")
    log.debug(f"returning node: {gymcts_node1}")
    return gymcts_node1


class DistributedGymctsAgent:
    render_tree_after_step: bool = False
    render_tree_max_depth: int = 2
    exclude_unvisited_nodes_from_render: bool = False
    number_of_simulations_per_step: int = 25
    num_parallel: int = 4

    env_ref: ObjectRef[GymctsABC]
    search_root_node_ref: ObjectRef[GymctsNode]  # NOTE: this is not the same as the root of the tree!
    clear_mcts_tree_after_step: bool

    def __init__(self,
                 env: GymctsABC,
                 clear_mcts_tree_after_step: bool = True,
                 render_tree_after_step: bool = False,
                 render_tree_max_depth: int = 2,
                 num_parallel: int = 4,
                 number_of_simulations_per_step: int = 25,
                 exclude_unvisited_nodes_from_render: bool = False
                 ):
        # check if action space of env is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Action space must be discrete.")

        self.num_parallel = num_parallel

        self.render_tree_after_step = render_tree_after_step
        self.exclude_unvisited_nodes_from_render = exclude_unvisited_nodes_from_render
        self.render_tree_max_depth = render_tree_max_depth

        self.number_of_simulations_per_step = number_of_simulations_per_step

        self.env_ref = ray.put(
            env
        )
        self.clear_mcts_tree_after_step = clear_mcts_tree_after_step

        self.search_root_node_ref = ray.put(GymctsNode(
            action=None,
            parent=None,
            env_reference=env,
        ))

    def solve(self, num_simulations_per_step: int = None, render_tree_after_step: bool = None) -> list[int]:

        if num_simulations_per_step is None:
            num_simulations_per_step = self.number_of_simulations_per_step
        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        log.debug(f"Solving from root node: {self.search_root_node_ref}")

        current_node_ref = self.search_root_node_ref

        action_list = []

        while not ray.get(current_node_ref).terminal:
            next_action, current_node_ref = self.perform_mcts_step(num_simulations=num_simulations_per_step,
                                                                   render_tree_after_step=render_tree_after_step)

            log.info(
                f"selected action {next_action} after {self.num_parallel} x {num_simulations_per_step} simulations.")
            action_list.append(next_action)
            log.info(f"current action list: {action_list}")

        log.info(f"Final action list: {action_list}")
        # restore state of current node
        return action_list

    def perform_mcts_step(self, search_start_node_ref: GymctsNode = None, num_simulations: int = None,
                          render_tree_after_step: bool = None, num_parallel: int = None) -> tuple[
        int, ObjectRef[GymctsNode]]:

        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        if render_tree_after_step is None:
            render_tree_after_step = self.render_tree_after_step

        if num_simulations is None:
            num_simulations = self.number_of_simulations_per_step

        if search_start_node_ref is None:
            search_start_node_ref = self.search_root_node_ref

        if num_parallel is None:
            num_parallel = self.num_parallel

        # action = self.vanilla_mcts_search(
        #    search_start_node=search_start_node,
        #   num_simulations=num_simulations,
        # )
        # next_node = search_start_node.children[action]
        print(self.env_ref)
        print(type(self.env_ref))
        mcts_interation_futures = [
            mcts_lookahead.remote(
                search_start_node_ref,
                self.env_ref,
                num_simulations
            )
            for _ in range(num_parallel)
        ]

        while mcts_interation_futures:
            ready_gymcts_nodes, mcts_interation_futures = ray.wait(mcts_interation_futures)
            for ready_node_ref in ready_gymcts_nodes:
                # merge the tree
                search_start_node_ref = merge_trees.remote(
                    search_start_node_ref,
                    ready_node_ref
                )

        search_start_node = ray.get(search_start_node_ref)
        action = search_start_node.get_best_action()
        next_node = search_start_node.children[action]

        if self.clear_mcts_tree_after_step:
            # to clear memory we need to remove all nodes except the current node
            # this is done by setting the root node to the current node
            # and setting the parent of the current node to None
            # we also need to reset the children of the current node
            # this is done by calling the reset method
            next_node.reset()

        next_node_ref = ray.put(next_node)

        return action, next_node_ref


if __name__ == '__main__':
    ray.init()

    log.setLevel(20)  # 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CR
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    # env1 = ActionHistoryMCTSGymEnvWrapper(env1)
    env = DeepCopyMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent1 = DistributedGymctsAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True,
        num_parallel=4,
    )
    import time

    start_time = time.perf_counter()
    actions = agent1.solve()
    end_time = time.perf_counter()

    print(f"solution time pro action: {end_time - start_time}/{len(actions)}")
