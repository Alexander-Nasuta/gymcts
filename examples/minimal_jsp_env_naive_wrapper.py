from collections import namedtuple
from copy import deepcopy, copy

from gymnasium.spaces import Discrete
from jsp_instance_utils.instances import ft06
from jsp_vis.console import gantt_chart_console

from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymnasium.wrappers import TransformReward, NormalizeReward
from gymcts.logger import log

import gymnasium as gym

import pandas as pd
import numpy as np

Operation = namedtuple("Operation", ["job_id", "op_id", "unique_op_id", "machine_type", "duration"])

import copy
import random


class JSPInstance:
    def __init__(
            self,
            jobs: list,
            num_ops_per_job: int = None,
            max_op_time: int = None,
            num_machines: int = None,
            id: str = None,
            opt_time: float = None,
            spt_time: float = None,
            intra_instance_op_entropy=None
    ):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job  # todo infer if not given
        self.max_op_time = max_op_time  # todo infer if not given
        self.num_machines = num_machines if num_machines else num_ops_per_job
        self.id = id
        self.spt_time = spt_time
        self.opt_time = opt_time
        self.intra_instance_op_entropy = intra_instance_op_entropy


def jsp_instance_adapter(jsp_instance):
    _, n_jobs, n_machines = jsp_instance.shape
    machine_order = jsp_instance[0]
    processing_times = jsp_instance[1]

    """
    Generates jobs consisting of operations with random durations and orders in which to be carried out,
    and returns a JSPInstance based on these jobs
    """
    jobs = []
    unique_op_id = 0
    max_op_duration = np.max(processing_times)
    for i in range(0, n_jobs):
        operations = []
        for j in range(0, n_machines):
            duration = processing_times[i, j]
            machine_type = machine_order[i, j]
            operations.append(Operation(i, j, unique_op_id, machine_type, duration))
            unique_op_id += 1

        jobs.append(operations)

    return JSPInstance(jobs, num_ops_per_job=n_machines, num_machines=n_machines,
                       max_op_time=max_op_duration)


class JobShopModel():
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_machines, max_duration=10):
        remaining_operations = []
        op_id = 0
        for j in range(num_jobs):
            job = []
            for m in range(num_machines):
                job.append(
                    Operation(j, m, op_id, random.randint(0, num_machines - 1), random.randint(0, max_duration - 1)))
                op_id += 1
            remaining_operations.append(job)

        schedule = [[] for i in range(num_machines)]

        last_job_ops = [-1 for _ in range(num_jobs)]
        return {'remaining_operations': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops}

    @staticmethod
    def _schedule_op(job_id, remaining_operations, schedule):
        possible = False

        if len(remaining_operations[job_id]) > 0:
            op = remaining_operations[job_id].pop(0)
            machine = op.machine_type
            start_time = JobShopModel._determine_start_time(op, schedule)
            schedule[machine].append((op, start_time, start_time + op.duration))
            possible = True
        return remaining_operations, schedule, possible

    @staticmethod
    def _schedule_op(job_id, remaining_operations, schedule, last_job_ops):
        possible = False

        if len(remaining_operations[job_id]) > 0:
            possible = True

            op = remaining_operations[job_id].pop(0)
            machine = op.machine_type
            start_time = JobShopModel._last_op_end(last_job_ops, op)
            machine_schedule = schedule[op.machine_type]
            if len(machine_schedule) == 0:
                schedule[machine].append((op, start_time, start_time + op.duration))
                last_job_ops[op.job_id] = start_time + op.duration
                return remaining_operations, schedule, last_job_ops, possible

            left_shift, left_shift_time, insertion_index = JobShopModel._left_shift_possible(start_time,
                                                                                             machine_schedule,
                                                                                             op.duration)
            if left_shift:
                schedule[machine].insert(insertion_index, (op, left_shift_time, left_shift_time + op.duration))
                new_time = left_shift_time + op.duration
                last_job_ops[op.job_id] = new_time if new_time > last_job_ops[op.job_id] else last_job_ops[op.job_id]

            else:
                last_op, start, end = machine_schedule[-1]

                if end > start_time:
                    start_time = end

                schedule[machine].append((op, start_time, start_time + op.duration))
                last_job_ops[op.job_id] = start_time + op.duration

        return remaining_operations, schedule, last_job_ops, possible

    @staticmethod
    def _left_shift_possible(earliest_start, machine_schedule, op_duration):
        if earliest_start < 0:
            earliest_start = 0

        last_end = earliest_start
        for index, (op, start_time, end_time) in enumerate(machine_schedule):
            if end_time < last_end:
                continue

            if (start_time - last_end) >= op_duration:
                return True, last_end, index

            last_end = end_time

        return False, -1, -1

    @staticmethod
    def _last_op_end(last_job_ops, op: Operation):
        start_time = 0

        if last_job_ops[op.job_id] > 0:
            start_time = last_job_ops[op.job_id]

        return start_time

    @staticmethod
    def _is_done(remaining_operations):
        for j in remaining_operations:
            if len(j) > 0: return False

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
    def step(state, action):
        remaining_ops, schedule, last_job_ops, possible = JobShopModel._schedule_op(action,
                                                                                    state['remaining_operations'],
                                                                                    state['schedule'],
                                                                                    state['last_job_ops'])

        reward = 0
        if not possible: reward = -1
        done = JobShopModel._is_done(remaining_ops)
        if done:
            reward = - JobShopModel._makespan(schedule)
        return {'remaining_operations': remaining_ops, 'schedule': schedule, 'last_job_ops': last_job_ops}, reward, done

    @staticmethod
    def legal_actions(state):
        return [job_id for job_id in range(len(state['remaining_operations'])) if
                len(state['remaining_operations'][job_id]) > 0]


class JobShopEnv(gym.Env):

    def __init__(self, jsp_instance: JSPInstance, **kwargs):
        self.model = JobShopModel()
        self._jsp_instance = jsp_instance
        self._initial_jsp_instance = copy.copy(jsp_instance)

        self.set_instance(instance=self._initial_jsp_instance)

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
            'remaining_operations': gym.spaces.Tuple([
                operation_space for _ in range(self._jsp_instance.num_jobs * self._jsp_instance.num_machines)
            ]),
            'last_job_ops': gym.spaces.Tuple([
                gym.spaces.Discrete(2) for _ in range(6)
            ]),
            'schedule': gym.spaces.Tuple([
                gym.spaces.Tuple([scheduled_operation_space for _ in range(self._jsp_instance.num_machines)]) for _ in
                range(self._jsp_instance.num_jobs)
            ])
        })
        self.observation_space = observation_space
        self.action_space = Discrete(6)

        self.reset()

    def set_instance(self, instance):
        self.done = False
        self.steps = 0
        self.instance = instance
        self.ops_per_job = self.instance.num_ops_per_job
        self.num_machines = self.instance.num_ops_per_job
        self.max_op_duration = self.instance.max_op_time
        self.num_jobs = self.instance.num_jobs

        schedule = [[] for _ in range(self.num_machines)]
        last_job_ops = [-1 for _ in range(self.num_jobs)]

        s_ = {'remaining_operations': deepcopy(self.instance.jobs), 'schedule': schedule,
              'last_job_ops': last_job_ops}

        self.state = s_
        return self.state

    def reset(self, **kwargs):
        self.done = False
        self.steps = 0
        self.set_instance(self._initial_jsp_instance)

        return self.state, {}

    def set_state(self, state):
        self.state = state
        if len(state['remaining_operations']) > 0:
            self.done = False

    def step(self, action):
        self.state, reward, self.done = self.model.step(self.state, action)
        self.steps += 1

        return self.state, reward, self.done, False, {}

    def render(self):
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
        num_of_machines = self._jsp_instance.num_machines
        gantt_chart_console(df, num_of_machines)
        print(f'Makespan: {latest_finish_time}')

    def raw_state(self):
        return self.state

    def current_instance(self):
        return self.instance

    def max_num_actions(self):
        return len(self.state['remaining_operations'])

    def current_num_steps(self) -> int:
        return self.steps

    def get_legal_action_mask(self) -> list[bool]:
        legal_action = self.model.legal_actions(self.state)
        legal_action_mask = [False for _ in range(self.action_space.n)]
        for action in legal_action:
            legal_action_mask[action] = True
        return legal_action_mask


if __name__ == '__main__':
    log.setLevel(20)

    mk_jsp_instance = jsp_instance_adapter(ft06)

    env = JobShopEnv(
        jsp_instance=mk_jsp_instance,
    )

    env.reset()
    env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
    env = TransformReward(env, lambda r: r / 36)


    def mask_fn(env: gym.Env) -> np.ndarray:
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.unwrapped.get_legal_action_mask()


    env = DeepCopyMCTSGymEnvWrapper(
        env,
        action_mask_fn=mask_fn
    )

    agent = GymctsAgent(
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
