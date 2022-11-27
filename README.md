# Gym-ANM
[![Downloads](https://pepy.tech/badge/gym-anm)](https://pepy.tech/project/gym-anm)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://gym-anm.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/robinhenry/gym-anm/branch/master/graph/badge.svg?token=7JSMJPPIQ7)](https://codecov.io/gh/robinhenry/gym-anm)
![CI (pip)](https://github.com/robinhenry/gym-anm/actions/workflows/ci_pip.yml/badge.svg)
![CI (conda)](https://github.com/robinhenry/gym-anm/actions/workflows/ci_conda.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`gym-anm` is a framework for designing reinforcement learning (RL) environments that model Active Network
Management (ANM) tasks in electricity distribution networks. It is built on top of the
[OpenAI Gym](https://github.com/openai/gym) toolkit.

The `gym-anm` framework was designed with one goal in mind: **bridge the gap between research in RL and in
the management of power systems**. We attempt to do this by providing RL researchers with an easy-to-work-with
library of environments that model decision-making tasks in power grids.

**Papers:**
*  [Gym-ANM: Reinforcement Learning Environments for Active Network Management Tasks in Electricity Distribution Systems](https://doi.org/10.1016/j.egyai.2021.100092)
*  [Gym-ANM: Open-source software to leverage reinforcement learning for power system management in research and education](https://doi.org/10.1016/j.simpa.2021.100092)

## Key features
*  Very little background in electricity systems modelling it required. This makes `gym-anm` an ideal starting point
   for RL students and researchers looking to enter the field.
*  The environments (tasks) generated by `gym-anm` follow the [OpenAI Gym](https://github.com/openai/gym)
   framework, with which a large part of the RL community is already familiar.
*  The flexibility of `gym-anm`, with its different customizable components, makes it a suitable framework
   to model a wide range of ANM tasks, from simple ones that can be used for educational purposes, to complex ones
   designed to conduct advanced research.

## Documentation
Documentation is provided online at [https://gym-anm.readthedocs.io/en/latest/](https://gym-anm.readthedocs.io/en/latest/).

## Installation

### Requirements
`gym-anm` requires Python 3.8+ and can run on Linux, MaxOS, and Windows.

We recommend installing `gym-anm` in a Python environment (e.g., [virtualenv](https://virtualenv.pypa.io/en/latest/)
or [conda](https://conda.io/en/latest/#)).

### Using pip
Using pip (preferably after activating your virtual environment):
```
pip install gym-anm
```

### Building from source
Alternatively, you can build `gym-anm` directly from source:
```
git clone https://github.com/robinhenry/gym-anm.git
cd gym-anm
pip install -e .
```

## Example
The following code snippet illustrates how `gym-anm` environments can be used. In this example,
actions are randomly sampled from the action space of the environment `ANM6Easy-v0`. For more information
about the agent-environment interface, see the official [OpenAI Gym documentation](https://github.com/openai/gym).
```
import gym
import time

def run():
   env = gym.make('gym_anm:ANM6Easy-v0')
   o = env.reset()

   for i in range(100):
       a = env.action_space.sample()
       o, r, done, info = env.step(a)
       env.render()
       time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.

   env.close()

if __name__ == '__main__':
    run()
```
The above code would render the environment in your default web browser as shown in the image below:
![alt text](https://github.com/robinhenry/gym-anm/blob/master/docs/source/images/anm6-easy-example.png?raw=true)

Additional example scripts can be found in [examples/](examples).

## Testing the installation
All unit tests in `gym-anm` can be ran from the project root directory with:
```
python -m pytest tests
```

## Contributing
Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## Citing the project
All publications derived from the use of `gym-anm` should cite the following two 2021 papers:
```
@article{HENRY2021100092,
    title = {Gym-ANM: Reinforcement learning environments for active network management tasks in electricity distribution systems},
    journal = {Energy and AI},
    volume = {5},
    pages = {100092},
    year = {2021},
    issn = {2666-5468},
    doi = {https://doi.org/10.1016/j.egyai.2021.100092},
    author = {Robin Henry and Damien Ernst},
}
```
```
@article{HENRY2021100092,
    title = {Gym-ANM: Open-source software to leverage reinforcement learning for power system management in research and education},
    journal = {Software Impacts},
    volume = {9},
    pages = {100092},
    year = {2021},
    issn = {2665-9638},
    doi = {https://doi.org/10.1016/j.simpa.2021.100092},
    author = {Robin Henry and Damien Ernst}
}
```

## Maintainers
`gym-anm` is currently maintained by [Robin Henry](https://www.robinxhenry.com/).

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
