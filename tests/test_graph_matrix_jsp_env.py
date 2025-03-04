from jsp_instance_utils.instances import ft06, ft06_makespan

from gymcts.gymcts_agent import GymctsAgent


def test_solution_quality_small_instance(graph_matrix_env_naive_wrapper_two_job_jsp_instance):
    env = graph_matrix_env_naive_wrapper_two_job_jsp_instance

    agent = GymctsAgent(env=env)

    actions = agent.solve(num_simulations_per_step=100)  # enough to do a full search
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    assert env.unwrapped.get_makespan() == 40


def test_solution_quality_ft06_with_():
    import gymnasium as gym
    import numpy as np

    from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
    from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper

    env = DisjunctiveGraphJspEnv(
        jsp_instance=ft06,
        reward_function="makespan",
        c_lb=-ft06_makespan,
    )
    env.reset()

    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.unwrapped.valid_action_mask()

    env = DeepCopyMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = GymctsAgent(
        env=env,
        number_of_simulations_per_step=200,
        clear_mcts_tree_after_step=False,
    )

    actions = agent.solve()

    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    assert term
    assert env.unwrapped.get_makespan() <= ft06_makespan * 1.5
