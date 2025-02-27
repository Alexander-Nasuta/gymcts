# Graph Matrix Job Shop Env

A Monte Carlo Tree Search Implementation for Gymnasium-style Environments.

- Github: [GYMCTS on Github](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv)
- Pypi: [GYMCTS on PyPi](https://pypi.org/project/graph-matrix-jsp-env/)
- Documentation: [GYMCTS Docs](https://graphmatrixjobshopenv.readthedocs.io/en/latest/)

## Description

This project provides a Monte Carlo Tree Search (MCTS) implementation for Gymnasium-style environments as an installable Python package.
The package is designed to be used with the Gymnasium interface.
It is especially useful for combinatorial optimization problems or planning problems, such as the Job Shop Scheduling Problem (JSP).
The documentation provides numerous examples on how to use the package with different environments, while focusing on scheduling problems.

A minimal working example is provided in the [Quickstart](#quickstart) section.

It comes with a variety of visualisation options, which is useful for research and debugging purposes. 
It aims to be a base for further research and development for neural guided search algorithms.
## Quickstart
To use the package, install it via pip:

```shell
pip install gymcts
```
The usage of a MCTS agent can roughly organised into the following steps:

- Create a Gymnasium-style environment
- Wrap the environment with a GymCTS wrapper
- Create a MCTS agent
- Solve the environment with the MCTS agent
- Render the solution

The GYMCTS package provides a two types of wrappers for Gymnasium-style environments:
- `NaiveSoloMCTSGymEnvWrapper`: A wrapper that uses deepcopies of the environment to save a snapshot of the environment state for each node in the MCTS tree.
- `RecordEpisodeStatistics`: A wrapper that records the episode statistics (e.g. episode length, episode return) and provides them in the info dictionary.

### FrozenLake Example

Here is a minimal example of how to use the package with the FrozenLake environment and the NaiveSoloMCTSGymEnvWrapper:
```python
import gymnasium as gym

from gymcts.gymcts_agent import SoloMCTSAgent
from gymcts.gymcts_naive_wrapper import NaiveSoloMCTSGymEnvWrapper

from gymcts.logger import log

# set log level to 20 (INFO) 
# set log level to 10 (DEBUG) to see more detailed information
log.setLevel(20)

if __name__ == '__main__':
    # 0. create the environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
    env.reset()

    # 1. wrap the environment with the naive wrapper or a custom gymcts wrapper
    env = NaiveSoloMCTSGymEnvWrapper(env)

    # 2. create the agent
    agent = SoloMCTSAgent(
        env=env,
        clear_mcts_tree_after_step=False,
        render_tree_after_step=True,
        number_of_simulations_per_step=50,
        exclude_unvisited_nodes_from_render=True
    )

    # 3. solve the environment
    actions = agent.solve()

    # 4. render the environment solution in the terminal
    print(env.render())
    for a in actions:
        obs, rew, term, trun, info = env.step(a)
        print(env.render())

    # 5. print the solution
    # read the solution from the info provided by the RecordEpisodeStatistics wrapper 
    # (that NaiveSoloMCTSGymEnvWrapper uses internally)
    episode_length = info["episode"]["l"]
    episode_return = info["episode"]["r"]

    if episode_return == 1.0:
        print(f"Environment solved in {episode_length} steps.")
    else:
        print(f"Environment not solved in {episode_length} steps.")
```

Here is a minimal example of how to use the package with the FrozenLake environment and the NaiveSoloMCTSGymEnvWrapper:

### Stable Baselines3 Example

To train a PPO agent using the environment with Stable Baselines3 one first needs to install the required dependencies:

```shell
pip install stable-baselines3
pip install sb3-contrib
```

Then one can use the following code to train a PPO agent:

```python

```

## Visualizations

The environment offers multiple visualisation options.
There are four visualisations that can be mixed and matched:
- `human` (default): prints a Gantt chart visualisation to the console.
- `ansi`: prints a visualisation of the graph matrix and the Gantt chart to the console.
- `debug`: prints a visualisation of the graph matrix. The debugs visualisation is maps the elements of the successor lists and unknown list to the original graph indices of the takes ad uses colors to separate the different elements. It also prints the Gantt chart and some additional information.
- `window`: creates a Gantt chart visualisation in a separate window.
- `rgb_array`: creates a Gantt chart visualisation as a RGB array. This mode return the RGB array of the `window` visualisation. This can be used to create a video of the Gantt chart visualisation. 

### Examples

For the following Job Shop Scheduling Problem (JSP) instance:

```python
from graph_matrix_jsp_env.disjunctive_jsp_env import DisjunctiveGraphJspEnv
import numpy as np

if __name__ == '__main__':
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ], dtype=np.int32)
    env = DisjunctiveGraphJspEnv(
        jsp_instance=custom_jsp_instance,
    )
    obs, info = env.reset()
    mode = 'debug' # replace with 'human', 'ansi', 'window', 'rgb_array' for different visualizations
    env.render(mode=mode) 

    for a in [5, 1, 2, 6, 3, 7, 4, 8]:
        env.step(a)
        env.render(mode=mode)

    env.render()
```

The individual rendering modes result in the following visualisations:

#### ANSI

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/asni-render.gif)

#### Debug

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/debug-render.gif)

#### Defualt (Human)

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/default-render.gif)

### window

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/window-render.gif)

The Terminal used for the visualisations is [Ghostty](https://github.com/ghostty-org/ghostty).

## State of the Project

This project is complementary material for a research paper. It will not be frequently updated.
Minor updates might occur.
Significant further development will most likely result in a new project. In that case, a note with a link will be added in the `README.md` of this project.  

## Dependencies

This project specifies multiple requirements files. 
`requirements.txt` contains the dependencies for the environment to work. These requirements will be installed automatically when installing the environment via `pip`.
`requirements_dev.txt` contains the dependencies for development purposes. It includes the dependencies for testing, linting, and building the project on top of the dependencies in `requirements.txt`.
`requirements_examples.txt` contains the dependencies for running the examples inside the project. It includes the dependencies in `requirements.txt` and additional dependencies for the examples.

In this Project the dependencies are specified in the `pyproject.toml` file with as little version constraints as possible.
The tool `pip-compile` translates the `pyproject.toml` file into a `requirements.txt` file with pinned versions. 
That way version conflicts can be avoided (as much as possible) and the project can be built in a reproducible way.

## Development Setup

If you want to check out the code and implement new features or fix bugs, you can set up the project as follows:

### Clone the Repository

clone the repository in your favorite code editor (for example PyCharm, VSCode, Neovim, etc.)

using https:
```shell
git clone https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv.git
```
or by using the GitHub CLI:
```shell
gh repo clone Alexander-Nasuta/GraphMatrixJobShopEnv
```

if you are using PyCharm, I recommend doing the following additional steps:

- mark the `src` folder as source root (by right-clicking on the folder and selecting `Mark Directory as` -> `Sources Root`)
- mark the `tests` folder as test root (by right-clicking on the folder and selecting `Mark Directory as` -> `Test Sources Root`)
- mark the `resources` folder as resources root (by right-clicking on the folder and selecting `Mark Directory as` -> `Resources Root`)

at the end your project structure should look like this:

todo

### Create a Virtual Environment (optional)

Most Developers use a virtual environment to manage the dependencies of their projects. 
I personally use `conda` for this purpose.

When using `conda`, you can create a new environment with the name 'my-graph-jsp-env' following command:

```shell
conda create -n my-graph-jsp-env python=3.11
```

Feel free to use any other name for the environment or an more recent version of python.
Activate the environment with the following command:

```shell
conda activate my-graph-jsp-env
```

Replace `my-graph-jsp-env` with the name of your environment, if you used a different name.

You can also use `venv` or `virtualenv` to create a virtual environment. In that case please refer to the respective documentation.

### Install the Dependencies

To install the dependencies for development purposes, run the following command:

```shell
pip install -r requirements_dev.txt
pip install tox
```

The testing package `tox` is not included in the `requirements_dev.txt` file, because it sometimes causes issues when 
using github actions. 
Github Actions uses an own tox environment (namely 'tox-gh-actions'), which can cause conflicts with the tox environment on your local machine.

Reference: [Automated Testing in Python with pytest, tox, and GitHub Actions](https://www.youtube.com/watch?v=DhUpxWjOhME).

### Install the Project in Editable Mode

To install the project in editable mode, run the following command:

```shell
pip install -e .
```

This will install the project in editable mode, so you can make changes to the code and test them immediately.

### Run the Tests

This project uses `pytest` for testing. To run the tests, run the following command:

```shell
pytest
```
Here is a screenshot of what the output might look like:

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/pytest-screenshot.png)

For testing with `tox` run the following command:

```shell
tox
```

Here is a screenshot of what the output might look like:

![](https://github.com/Alexander-Nasuta/GraphMatrixJobShopEnv/raw/master/resources/tox-screenshot.png)

Tox will run the tests in a separate environment and will also check if the requirements are installed correctly.

### Builing and Publishing the Project to PyPi 

In order to publish the project to PyPi, the project needs to be built and then uploaded to PyPi.

To build the project, run the following command:

```shell
python -m build
```

It is considered good practice use the tool `twine` for checking the build and uploading the project to PyPi.
By default the build command creates a `dist` folder with the built project files.
To check all the files in the `dist` folder, run the following command:

```shell
twine check dist/**
```

If the check is successful, you can upload the project to PyPi with the following command:

```shell
twine upload dist/**
```

### Documentation
This project uses `sphinx` for generating the documentation. 
It also uses a lot of sphinx extensions to make the documentation more readable and interactive.
For example the extension `myst-parser` is used to enable markdown support in the documentation (instead of the usual .rst-files).
It also uses the `sphinx-autobuild` extension to automatically rebuild the documentation when changes are made.
By running the following command, the documentation will be automatically built and served, when changes are made (make sure to run this command in the root directory of the project):

```shell
sphinx-autobuild ./docs/source/ ./docs/build/html/
```

This project features most of the extensions featured in this Tutorial: [Document Your Scientific Project With Markdown, Sphinx, and Read the Docs | PyData Global 2021](https://www.youtube.com/watch?v=qRSb299awB0).



## Contact

If you have any questions or feedback, feel free to contact me via [email](mailto:alexander.nasuta@wzl-iqs.rwth-aachen.de) or open an issue on repository.
