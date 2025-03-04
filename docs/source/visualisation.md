# Visualizations

The MCTS agent provides a visualisation of the MCTS tree.
Below is an example code snippet that shows how to use the visualisation options of the MCTS agent.

The following metrics are displayed in the visualisation:
- `N`: the number of visits of the node
- `Q_v`: the average return of the node
- `ubc`: the upper confidence bound of the node
- `a`: the action that leads to the node
- `best`: the highest return of any rollout from the node

`Q_v` and `ubc` have a color gradient from red to green, where red indicates a low value and green indicates a high value.
The color gradient is based on the minimum and maximum values of the respective metric in the tree.

The visualisation is rendered in the terminal and can be limited to a certain depth of the tree.
The default depth is 2.

```python
import gymnasium as gym

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_action_history_wrapper import ActionHistoryMCTSGymEnvWrapper
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

from gymcts.logger import log

# set log level to 20 (INFO)
# set log level to 10 (DEBUG) to see more detailed information
log.setLevel(20)

if __name__ == '__main__':
    # create the environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")
    env.reset()

    # wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = ActionHistoryMCTSGymEnvWrapper(env)

    # create the agent
    agent = GymctsAgent(
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
```