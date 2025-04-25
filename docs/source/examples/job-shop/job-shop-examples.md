# Job Shop Gallery

The Job Shop Scheduling Problem (JSP) is a classic optimization problem in operations research and computer science.
It involves scheduling a set of jobs on a set of machines, where each job consists of a sequence of operations that must be performed in a specific order.
The goal is to minimize the total time required to complete all jobs, known as the makespan.

For more information about the Job Shop Scheduling Problem, fell free to check out the Documentation page of [`graph-jsp-env`](https://graph-jsp-env.readthedocs.io/en/latest/theory/theory.html).

This section contains a collection of examples that demonstrate how to use the library with the Job Shop Scheduling Problem with different environments.
This documentation includes the following environments:
- [`graph-jsp-env`](https://graph-jsp-env.readthedocs.io/en/latest/): A graph-based environment for the Job Shop Scheduling Problem.
- [`graph-matrix-jsp-env`](https://graphmatrixjobshopenv.readthedocs.io/en/latest/): A matrix-based environment for the Job Shop Scheduling Problem.
- [`JSSEnv`](https://github.com/prosysscience/JSSEnv): A custom environment for the Job Shop Scheduling Problem.
- [`gnn_jsp_env`](https://git.rwth-aachen.de/jobshop/neuralmcts): A graph neural network-based environment for the Job Shop Scheduling Problem.
- [`minimal_jsp_env`](https://git.rwth-aachen.de/jobshop/neuralmcts): A minimal environment for the Job Shop Scheduling Problem.

The example are given as jupyter notebooks, so you can see the expected output and the code used to generate it.

```{note}
The JSSEnv is originally gym environment (not a gymnasium environment).
Therfore it was adapted to the gymnasium standard in the examples.
```

```{note}
The `gnn_jsp_env` and `minimal_jsp_env` are not available on PyPI as stand alone packages. 
Therfore they were copied into the examples.
```

```{note}
The example all have the visualization of the MCTS tree enabled.
This is done to show the tree structure of the MCTS algorithm and how it evolves over time.

In a productive environment, you might want to disable the visualization of the MCTS tree for more compute efficiency.
```




```{toctree}
:maxdepth: 2

graph-jsp-env_deepcopy_wrapper.ipynb
graph-jsp-env_action_history_wrapper.ipynb
graph-jsp-env_custom_wrapper.ipynb

graph-matrix-jsp-env_deepcopy_wrapper.ipynb
graph-matrix-jsp-env_action_history_wrapper.ipynb
graph-matrix-jsp-env_custom_wrapper.ipynb

JSSEnv_action_history_wrapper.ipynb
JSSEnv_deepcopy_wrapper.ipynb

gnn_jsp_env_deepcopy_wrapper.ipynb
gnn_jsp_env_action_history_wrapper.ipynb

minimal_jsp_env_action_history_wrapper.ipynb
minimal_jsp_env_deepcopy_wrapper.ipynb
```
