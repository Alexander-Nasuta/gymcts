


def estimate_cost(n_jobs, n_machines, n_sims_per_step) -> int:
    episode_length = n_jobs * n_machines

    upper_estimate = 0
    lower_estimate = 0
    for steps_within_episode in range(episode_length):
        upper_estimate += steps_within_episode * n_sims_per_step
        lower_estimate += steps_within_episode * 0.9 * n_sims_per_step


    return upper_estimate, lower_estimate




if __name__ == '__main__':
    n_jobs = 10
    n_machines = 10
    n_sims_per_step = 900

    upper_cost, lower_cost = estimate_cost(
        n_jobs=n_jobs,
        n_machines=n_machines,
        n_sims_per_step=n_sims_per_step
    )
    print(f"Estimated upper cost: {upper_cost}")
    print(f"Estimated lower cost: {lower_cost}")
