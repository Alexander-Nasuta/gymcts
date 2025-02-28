import gymnasium as gym

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_deterministic_wrapper import DeterministicSoloMCTSGymEnvWrapper
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

# set log level to 20 (INFO)
# set log level to 10 (DEBUG) to see more detailed information
log.setLevel(20)

if __name__ == '__main__':
    # create the environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")
    env.reset()

    # wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = DeterministicSoloMCTSGymEnvWrapper(env)

    # create the agent
    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=False,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True,  # weather to exclude unvisited nodes from the render
        render_tree_max_depth=2  # the maximum depth of the tree to render
    )

    # solve the environment
    actions = agent.solve()

    # render the MCTS tree from the root
    # search_root_node is the node that corresponds to the current state of the environment in the search process
    # since we called agent.solve() we are at the end of the search process
    log.info(f"MCTS Tree starting at the final state of the environment (actions: {agent.search_root_node.state})")
    agent.show_mcts_tree(
        start_node=agent.search_root_node,
    )

    # the parent of the terminal node (which we are rendering below) is the search root node of the previous step in the
    # MCTS solving process
    log.info(
        f"MCTS Tree starting at the pre-final state of the environment (actions: {agent.search_root_node.parent.state})")
    agent.show_mcts_tree(
        start_node=agent.search_root_node.parent,
    )

    # render the MCTS tree from the root
    log.info(f"MCTS Tree starting at the root state (actions: {agent.search_root_node.get_root().state})")
    agent.show_mcts_tree(
        start_node=agent.search_root_node.get_root(),
        # you can limit the depth of the tree to render to any number
        tree_max_depth=1
    )
