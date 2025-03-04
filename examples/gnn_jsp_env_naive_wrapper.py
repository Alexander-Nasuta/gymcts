from collections import namedtuple
from copy import deepcopy, copy

from gymnasium.spaces import Box, Discrete
from jsp_instance_utils.instances import ft06, ft06_makespan
from jsp_vis.console import gantt_chart_console

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_deterministic_wrapper import DeterministicSoloMCTSGymEnvWrapper
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward, NormalizeReward
from gymcts.logger import log

import gymnasium as gym

from gymnasium.core import ActType, ObsType
from typing import Any, SupportsFloat

import random

import pandas as pd
import numpy as np

Operation = namedtuple("Operation", ["job_id", "op_id", "unique_op_id", "machine_type", "duration"])


def get_legal_pos(op_dur, job_ready_time, possible_pos, mch_infos):
    """
    Returns the positions which fit the given operation duration,
    considering that the operation can only start when the required machine is free
    and the job is ready (the previous operations of the job are completed)
    """
    earliest_start_time = max(job_ready_time, mch_infos['end_times'][possible_pos[0] - 1])
    possible_pos_end_times = np.append(earliest_start_time, mch_infos['end_times'][possible_pos])[:-1]
    possible_gaps = mch_infos['start_times'][possible_pos] - possible_pos_end_times
    legal_pos_idx = np.where(op_dur <= possible_gaps)[0]
    legal_pos = np.take(possible_pos, legal_pos_idx)
    return legal_pos_idx, legal_pos, possible_pos_end_times


def put_in_the_end(op, job_ready_time, mch_ready_time, mch_infos):
    """
    Puts an operation at the end of the already scheduled operations
    """
    index = np.where(mch_infos['start_times'] == -1)[0][0]
    op_start_time = max(job_ready_time, mch_ready_time)
    mch_infos['op_ids'][index] = op.unique_op_id
    mch_infos['start_times'][index] = op_start_time
    mch_infos['end_times'][index] = op_start_time + op.duration
    return op_start_time


def put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times, mch_infos):
    """
    Puts an operation between already scheduled operations
    """
    earliest_idx = legal_pos_idx[0]
    earliest_pos = legal_pos[0]
    start_time = possible_pos_end_times[earliest_idx]
    mch_infos['op_ids'][:] = np.insert(mch_infos['op_ids'], earliest_pos, op.unique_op_id)[:-1]
    mch_infos['start_times'][:] = np.insert(mch_infos['start_times'], earliest_pos, start_time)[:-1]
    mch_infos['end_times'][:] = np.insert(mch_infos['end_times'], earliest_pos, start_time + op.duration)[:-1]
    return start_time


def get_end_time_lbs(jobs, machine_infos):
    """
    Calculates the end time lower bounds for all operations
    :param jobs: array if jobs, where each job is an array of operations
    :param machine_infos: dictionary where the keys are machine indices and the values contain
    the ids of the operations scheduled on the machine (in the scheduled order), and the
    corresponding start and end times
    :returns: np array containing the end time lower bounds of all operations
    """
    end_times = [m['end_times'][i] for m in machine_infos.values() for i in range(len(m['end_times']))]
    op_ids = [m['op_ids'][i] for m in machine_infos.values() for i in range(len(m['op_ids']))]
    lbs = -1 * np.ones((len(jobs), len(jobs[0])))

    for i, job in enumerate(jobs):
        for j, op in enumerate(job):
            if op.unique_op_id in op_ids:
                lbs[i][j] = end_times[op_ids.index(op.unique_op_id)]
            elif j > 0:
                lbs[i][j] = lbs[i][j - 1] + op.duration
            else:
                lbs[i][j] = op.duration

    return lbs


def get_op_nbghs(op, machine_infos):
    """
    Finds a given operation's predecessor and successor on the machine where the operation is carried out
    """
    for key, value in machine_infos.items():
        if op.unique_op_id in value['op_ids']:
            action_coord = [key, np.where(op.unique_op_id == value['op_ids'])[0][0]]
            break
    assert action_coord, "The operation's unique id was not found in the machine informations"

    if action_coord[1].item() > 0:
        pred_id = action_coord[0], action_coord[1] - 1
    else:
        pred_id = action_coord[0], action_coord[1]
    pred = machine_infos[pred_id[0]]['op_ids'][pred_id[1]]

    if action_coord[1].item() + 1 < machine_infos[action_coord[0]]['op_ids'].shape[-1]:
        succ_temp_id = action_coord[0], action_coord[1] + 1
    else:
        succ_temp_id = action_coord[0], action_coord[1]
    succ_temp = machine_infos[succ_temp_id[0]]['op_ids'][succ_temp_id[1]]
    succ = op.unique_op_id if succ_temp < 0 else succ_temp

    return pred, succ


def get_first_ops(state):
    """
    Returns an array containing the unique indices of the first operations of each job.
    """
    num_ops = len(state['features'])
    num_jobs = len(state['jobs'])
    first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
    return first_col


class GNNJobShopModel():
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_ops_per_job, num_machines, max_duration=10):
        remaining_operations = []
        unique_op_id = 0
        for i in range(num_jobs):
            job = []
            for j in range(num_ops_per_job):
                job.append(Operation(i, j, unique_op_id, random.randint(0, num_machines - 1),
                                     random.randint(0, max_duration - 1)))
                unique_op_id += 1

            remaining_operations.append(job)

        schedule = [[] for _ in range(num_machines)]

        num_ops = num_jobs * num_machines

        # Number of operations scheduled on each machine
        ops_per_machine = [len([op for job in remaining_operations for op in job if op.machine_type == m]) for m in
                           range(num_machines)]
        # Information for each machine: the ids of the operations scheduled on it (in the scheduled order), and the
        # corresponding start and end times
        machine_infos = {m: {'op_ids': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'start_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'end_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32)} for m in
                         range(num_machines)}
        # Time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(num_jobs)]
        # Time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(num_machines)]

        jobs = deepcopy(remaining_operations)
        adj_matrix = GNNJobShopModel.init_adj_matrix(num_ops, num_jobs)
        features = GNNJobShopModel.init_features(jobs)

        node_states = np.array([1 if i % num_ops_per_job == 0 else 0 for i in range(num_ops)],
                               dtype=np.single)

        return {'remaining_ops': remaining_operations, 'schedule': schedule, 'machine_infos': machine_infos,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'node_states': node_states, 'jobs': jobs}

    @staticmethod
    def _schedule_op(job_id, state):
        possible = False

        if len(state['remaining_ops'][job_id]) > 0:
            op = state['remaining_ops'][job_id].pop(0)
            start_time, flag = GNNJobShopModel._determine_start_time(op, state['last_job_ops'],
                                                                     state['last_mch_ops'], state['machine_infos'])
            # Insert the operation at the correct position so that the entries remain sorted according to start_time
            state['schedule'][op.machine_type].append((op, start_time, start_time + op.duration))
            state['schedule'][op.machine_type] = sorted(state['schedule'][op.machine_type], key=lambda x: x[1])

            # Update state
            if state['last_job_ops'][op.job_id] < start_time + op.duration:
                state['last_job_ops'][op.job_id] = start_time + op.duration
            if state['last_mch_ops'][op.machine_type] < start_time + op.duration:
                state['last_mch_ops'][op.machine_type] = start_time + op.duration
            GNNJobShopModel._update_adj_matrix(state, op, flag)
            GNNJobShopModel._update_features(state, op)
            GNNJobShopModel._update_node_states(state, op)

            possible = True

        return state, possible

    @staticmethod
    def _update_adj_matrix(state, op, flag):
        # Update the adjacency matrix after a new operation has been scheduled
        pred, succ = get_op_nbghs(op, state['machine_infos'])
        state['adj_matrix'][op.unique_op_id] = 0
        state['adj_matrix'][op.unique_op_id, op.unique_op_id] = 1
        state['adj_matrix'][op.unique_op_id, pred] = 1
        state['adj_matrix'][succ, op.unique_op_id] = 1
        if op.unique_op_id not in get_first_ops(state):
            state['adj_matrix'][op.unique_op_id, op.unique_op_id - 1] = 1
        # Remove the old arc when a new operation inserts between two operations
        if flag and pred != op.unique_op_id and succ != op.unique_op_id:
            state['adj_matrix'][succ, pred] = 0

    @staticmethod
    def _update_features(state, op):
        # Update the operations' features after a new operation has been scheduled
        lower_bounds = get_end_time_lbs(state['jobs'], state['machine_infos'])  # recalculate lower bounds
        finished = np.array([f[1] if i != op.unique_op_id
                             else 1 for i, f in enumerate(state['features'])])  # set op as finished
        assert norm_coeff > 0, "The normalization coefficient has not been initialized"

        state['features'] = np.concatenate((lower_bounds.reshape(-1, 1) / norm_coeff,
                                            finished.reshape(-1, 1)), axis=1)

    @staticmethod
    def _update_node_states(state, op):
        succ = op.unique_op_id + 1 if ((op.unique_op_id + 1) % len(state['jobs'][0]) != 0) else op.unique_op_id
        if succ != op.unique_op_id:
            state['node_states'][op.unique_op_id] = 0  # TODO node_states type changes -> fix
            state['node_states'][succ] = 1  # TODO add -1 condition?

    @staticmethod
    def _determine_start_time(op: Operation, last_job_ops, last_mch_ops, machine_infos):
        job_ready_time = last_job_ops[op.job_id] if last_job_ops[op.job_id] != -1 else 0
        mch_ready_time = last_mch_ops[op.machine_type] if last_mch_ops[op.machine_type] != -1 else 0
        # Whether the operation is scheduled between already scheduled operations (True) or in the end (False)
        flag = False

        # Positions between already scheduled operations on the machine required by the operation
        possible_pos = np.where(job_ready_time < machine_infos[op.machine_type]['start_times'])[0]

        if len(possible_pos) == 0:
            # Not possible to schedule the operation between other operations -> put in the end
            op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, machine_infos[op.machine_type])
        else:
            # Positions which fit the length of the operation (there is enough time before the next operation)
            legal_pos_idx, legal_pos, possible_pos_end_times = get_legal_pos(op.duration, job_ready_time,
                                                                             possible_pos,
                                                                             machine_infos[op.machine_type])
            if len(legal_pos) == 0:
                # No position which can fit the operation -> put in the end
                op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, machine_infos[op.machine_type])
            else:
                # Schedule the operation between other operations
                op_start_time = put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times,
                                               machine_infos[op.machine_type])
                flag = True

        return op_start_time, flag

    @staticmethod
    def _is_done(remaining_ops):
        for j in remaining_ops:
            if len(j) > 0:
                return False

        return True

    @staticmethod
    def _makespan(schedule):
        makespan = 0

        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                _, _, end_time = machine_schedule[-1]
                if end_time > makespan:
                    makespan = end_time

        return makespan

    @staticmethod
    def _get_norm_coeff(max_duration, num_ops_per_job, num_jobs):
        i = 10
        while i < max_duration * num_ops_per_job * num_jobs:
            i *= 10
        return i

    @staticmethod
    def step(state, action):
        new_state, possible = GNNJobShopModel._schedule_op(action, deepcopy(state))

        reward = 0
        if not possible:
            reward = -1
        done = GNNJobShopModel._is_done(new_state['remaining_ops'])
        if done:
            reward = - GNNJobShopModel._makespan(new_state['schedule'])

        return new_state, reward, done

    @staticmethod
    def legal_actions(state):
        return [job_id for job_id in range(len(state['remaining_ops'])) if
                len(state['remaining_ops'][job_id]) > 0]

    @staticmethod
    def init_adj_matrix(num_ops, num_jobs):
        # task ids for first column (array containing the first tasks for each job)
        first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
        # task ids for last column (array containing the last tasks for each job)
        last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, -1]

        # conjunctive arcs showing precedence relations between tasks of the same job
        # np array with 1s on the row above the main diagonal and 0s everywhere else
        conj_nei_up_stream = np.eye(num_ops, k=-1, dtype=np.single)
        # np array with 1s on the row below the main diagonal and 0s everywhere else
        conj_nei_low_stream = np.eye(num_ops, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[last_col] = 0

        # self edges for all nodes
        # np array with 1s on the main diagonal and 0s everywhere else
        self_as_nei = np.eye(num_ops, dtype=np.single)

        adj = self_as_nei + conj_nei_up_stream
        return adj

    @staticmethod
    def init_features(jobs):
        durations = np.array([[op.duration for op in job] for job in jobs])
        lower_bounds = np.cumsum(durations, axis=1, dtype=np.single)  # lower bounds of operations' completion times
        machine_types = np.array([[op.machine_type for op in job] for job in jobs])
        finished_mark = np.zeros_like(machine_types, dtype=np.single)  # 0 for unfinished, 1 for finished
        global norm_coeff
        norm_coeff = GNNJobShopModel._get_norm_coeff(max(durations.flatten()), len(jobs[0]), len(jobs))

        # node features: normalized end time lower bounds and binary indicator of whether the action has been scheduled
        features = np.concatenate((lower_bounds.reshape(-1, 1) / norm_coeff,  # normalize the lower bounds
                                   finished_mark.reshape(-1, 1)), axis=1)  # 1 if scheduled, 0 otherwise

        return features


class GNNJobShopModelEnv(gym.Env):


    def _jsp_instance_adapter(self, jsp_instance):
        _, n_jobs, n_machines = jsp_instance.shape
        machine_order = jsp_instance[0]
        processing_times = jsp_instance[1]

        remaining_operations = []
        unique_op_id = 0

        for i in range(n_jobs):
            job = []
            num_ops_per_job = n_machines
            for j in range(num_ops_per_job):
                job.append(
                    Operation(
                        i, j,
                        unique_op_id,
                        machine_order[i][j], # machine_type
                        processing_times[i][j] # duration
                    )
                )
                unique_op_id += 1

            remaining_operations.append(job)

        schedule = [[] for _ in range(n_machines)]

        num_ops = n_jobs * n_machines

        # Number of operations scheduled on each machine
        ops_per_machine = [len([op for job in remaining_operations for op in job if op.machine_type == m]) for m in
                           range(n_machines)]
        # Information for each machine: the ids of the operations scheduled on it (in the scheduled order), and the
        # corresponding start and end times
        machine_infos = {m: {'op_ids': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'start_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'end_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32)} for m in
                         range(n_machines)}
        # Time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(n_jobs)]
        # Time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(n_machines)]

        jobs = deepcopy(remaining_operations)
        adj_matrix = GNNJobShopModel.init_adj_matrix(num_ops, n_jobs)
        features = GNNJobShopModel.init_features(jobs)

        node_states = np.array([1 if i % num_ops_per_job == 0 else 0 for i in range(num_ops)],
                               dtype=np.single)

        return {'remaining_ops': remaining_operations, 'schedule': schedule, 'machine_infos': machine_infos,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'node_states': node_states, 'jobs': jobs}

    def __init__(self, jsp_instance:np.array, **kwargs):
        self.model = GNNJobShopModel()


        _, n_jobs, n_machines = jsp_instance.shape

        self.n_jobs = n_jobs
        # self.n_ops_per_job = 6
        self.n_machines = n_machines

        # self.state = self.model.random_problem(6, 6, 6)
        self.state = self._jsp_instance_adapter(jsp_instance)
        self._initial_state = copy(self.state)

        # Define the space for an Operation
        operation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(np.iinfo(np.int32).max),  # job_id
            gym.spaces.Discrete(np.iinfo(np.int32).max),  # op_id
            gym.spaces.Discrete(np.iinfo(np.int32).max),  # unique_op_id
            gym.spaces.Discrete(np.iinfo(np.int32).max),  # machine_type
            gym.spaces.Discrete(np.iinfo(np.int32).max)  # duration
        ))

        # Define the space for a ScheduledOperation
        scheduled_operation_space = gym.spaces.Tuple((
            operation_space,  # Operation
            gym.spaces.Discrete(np.iinfo(np.int32).max),  # start_time
            gym.spaces.Discrete(np.iinfo(np.int32).max)  # end_time
        ))

        observation_space = gym.spaces.Dict({
            'adj_matrix': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'features': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'last_job_ops': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'last_mch_ops': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'machine_infos': gym.spaces.Dict({
                key_idx: gym.spaces.Dict({
                    'end_times': Box(low=-1, high=np.iinfo(np.int32).max, shape=(5,), dtype=np.int32),
                    'op_ids': Box(low=-1, high=np.iinfo(np.int32).max, shape=(5,), dtype=np.int32),
                    'start_times': Box(low=-1, high=np.iinfo(np.int32).max, shape=(5,), dtype=np.int32),
                }) for key_idx in range(6)
            }),
            # 'remaining_ops': ,
            'schedule': gym.spaces.Tuple([
                gym.spaces.Tuple([scheduled_operation_space for _ in range(self.n_machines)]) for _ in
                range(self.n_jobs)
            ])
        })
        self.observation_space = observation_space
        self.action_space = Discrete(6)

        self.done = False

    def set_state(self, state: dict):
        self.state = state
        if len(state['remaining_ops']) > 0:
            self.done = False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.state, reward, self.done = self.model.step(self.state, action)
        return self.state, reward, self.done, False, {'makespan': - reward}

    def render(self) -> None:
        allocation = []
        latest_finish_time = 0
        for mache_ops in self.state['schedule']:
            if mache_ops and len(mache_ops):

                for ops_elem, start_time, finish_time in mache_ops:
                    entry = {
                        'Task': f'Job {ops_elem.job_id}',
                        'Start': start_time,
                        'Finish': finish_time,
                        'Resource': f'Machine {ops_elem.machine_type}'
                    }
                    latest_finish_time = max(finish_time, latest_finish_time)
                    allocation.append(entry)

        df = pd.DataFrame(allocation)
        num_of_machines = self.n_machines
        gantt_chart_console(df, num_of_machines)
        print(f'Makespan: {latest_finish_time}')

    def get_state(self) -> dict:
        return self.state

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.set_state(self._initial_state)
        return self.state, {}

    def get_legal_action_mask(self) -> list[bool]:
        legal_action = self.model.legal_actions(self.state)
        legal_action_mask = [False for _ in range(self.action_space.n)]
        for action in legal_action:
            legal_action_mask[action] = True
        return legal_action_mask


if __name__ == '__main__':
    log.setLevel(20)

    # model = GNNJobShopModel()
    # jsp_state = model.random_problem(6, 6, 6)
    # print(pprint.pformat(jsp_state))

    env = GNNJobShopModelEnv(
        jsp_instance=ft06,
    )

    env.reset()
    env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
    env = TransformReward(env, lambda r: r / 36)

    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.get_legal_action_mask()


    env = NaiveSoloMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        exclude_unvisited_nodes_from_render=True,
        number_of_simulations_per_step=125,
    )

    root = agent.search_root_node.get_root()

    actions = agent.solve(render_tree_after_step=True)

    env.reset()
    for a in actions:
        obs, rew, term, trun, info = env.step(a)

    env.unwrapped.render()
