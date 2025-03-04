from gymcts.gymcts_agent import GymctsAgent


def test_number_of_visits_without_clearing_root(graph_matrix_env_naive_wrapper_singe_job_jsp_instance):
    env = graph_matrix_env_naive_wrapper_singe_job_jsp_instance

    agent = GymctsAgent(env=env, clear_mcts_tree_after_step=False)

    assert agent.search_root_node.visit_count == 0
    agent.vanilla_mcts_search(search_start_node=agent.search_root_node, num_simulations=10)

    assert agent.search_root_node.visit_count == 10

    agent.vanilla_mcts_search(search_start_node=agent.search_root_node, num_simulations=15)

    assert agent.search_root_node.visit_count == 25



def test_number_of_visits_without_clearing(graph_matrix_env_naive_wrapper_singe_job_jsp_instance):
    env = graph_matrix_env_naive_wrapper_singe_job_jsp_instance

    agent = GymctsAgent(env=env, clear_mcts_tree_after_step=False)
    assert agent.search_root_node.visit_count == 0

    actions = agent.solve(num_simulations_per_step=10)

    assert len(actions) == 4

    assert agent.search_root_node.get_root().visit_count == 10 * 4

    # check visit counts for all nodes
    for node, counts, tree_depth in zip(
            agent.search_root_node.get_root().traverse_nodes(),
            [40, 39, 38, 37],
            [4, 3, 2, 1]
    ):
        assert node.visit_count == counts
        assert node.max_tree_depth() == tree_depth


def test_number_of_visits_without_clearing_root_dynamic_step_size(graph_matrix_env_naive_wrapper_singe_job_jsp_instance):

    env = graph_matrix_env_naive_wrapper_singe_job_jsp_instance

    agent = GymctsAgent(env=env, clear_mcts_tree_after_step=False)

    tree_root = agent.search_root_node

    assert agent.search_root_node.visit_count == 0
    agent.perform_mcts_step(search_start_node=agent.search_root_node, num_simulations=10)

    assert agent.search_root_node.visit_count == 10 - 1 # -1 because the root node is not visited one before expanding
    assert agent.search_root_node.get_root() == tree_root

    agent.perform_mcts_step(search_start_node=agent.search_root_node, num_simulations=15)

    assert agent.search_root_node.visit_count == (10 - 1) + 15 - 1
    assert agent.search_root_node.get_root() == tree_root


def test_number_of_visits_with_clearing_root_dynamic_step_size(graph_matrix_env_naive_wrapper_singe_job_jsp_instance):

    env = graph_matrix_env_naive_wrapper_singe_job_jsp_instance

    agent = GymctsAgent(env=env, clear_mcts_tree_after_step=True)

    assert agent.search_root_node.visit_count == 0
    action, node = agent.perform_mcts_step(search_start_node=agent.search_root_node, num_simulations=10)
    assert agent.search_root_node == node
    assert agent.search_root_node.is_root()

    assert agent.search_root_node.visit_count == 0

    action, node = agent.perform_mcts_step(search_start_node=agent.search_root_node, num_simulations=15)
    assert agent.search_root_node.visit_count == 0
    assert agent.search_root_node == node
    assert agent.search_root_node.is_root()




def test_number_of_visits_with_clearing_root2(graph_matrix_env_naive_wrapper_two_job_jsp_instance):
    env = graph_matrix_env_naive_wrapper_two_job_jsp_instance

    steps = 50 # total of step to get an exhaustive tree is 2**4+1 = 16 +1 = 17
    # this means after the first step the tree should not grow anymore

    agent = GymctsAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        number_of_simulations_per_step=50
    )


    for i in range(8):
        action, node = agent.perform_mcts_step()
        assert node.get_root().max_tree_depth() == 8
        assert node.get_root().visit_count == (i+1) * 50

    assert node.terminal








