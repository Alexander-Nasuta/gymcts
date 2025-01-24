import pytest
import numpy as np
import gymnasium as gym


@pytest.fixture(scope="function")
def single_job_jsp_instance():
    single_job_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            # [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            # [5, 16, 7, 4]  # task durations of job 1
        ]
    ], dtype=np.int32)
    yield single_job_instance

@pytest.fixture(scope="function")
def two_job_jsp_instance():
    two_job_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]
    ], dtype=np.int32)

    yield two_job_jsp_instance


@pytest.fixture(scope="function")
def graph_matrix_env_naive_wrapper_singe_job_jsp_instance(single_job_jsp_instance):


    from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
    from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

    env = DisjunctiveGraphJspEnv(
        jsp_instance=single_job_jsp_instance,
    )
    env.reset()

    def mask_fn(env: gym.Env) -> np.ndarray:

        return env.unwrapped.valid_action_mask()

    env = NaiveSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    yield env


@pytest.fixture(scope="function")
def graph_matrix_env_naive_wrapper_two_job_jsp_instance(two_job_jsp_instance):


    from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
    from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

    env = DisjunctiveGraphJspEnv(jsp_instance=two_job_jsp_instance)
    env.reset()

    def mask_fn(env: gym.Env) -> np.ndarray:

        return env.unwrapped.valid_action_mask()

    env = NaiveSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    yield env





