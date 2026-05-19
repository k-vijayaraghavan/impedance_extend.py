![GitHub release](https://img.shields.io/github/release/k-vijayaraghavan/impedance_extend.py)

![PyPI - Downloads](https://img.shields.io/pypi/dm/impedance_extend?style=flat-square)  [![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors)

[![Build Status](https://github.com/k-vijayaraghavan/impedance_extend.py/actions/workflows/ci.yml/badge.svg)](https://github.com/k-vijayaraghavan/impedance_extend.py/actions)  [![Documentation Status](https://readthedocs.org/projects/impedance-extendpy/badge/?version=latest&kill_cache=1)](https://impedance-extendpy.readthedocs.io/en/latest/?badge=latest) [![Coverage Status](https://coveralls.io/repos/github/ECSHackWeek/impedance.py/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/ECSHackWeek/impedance.py?branch=master)

impedance_extend.py
------------

`impedance_extend.py` is a Python package for making electrochemical impedance spectroscopy (EIS) analysis reproducible and easy-to-use. It extends [impedance.py](https://github.com/ECSHackWeek/impedance.py) by adding least_squares, GA (using pygad), and PSO (using pyswarms) as additional optimization methods. `impedance_extend.py` additionally supports sequential optimization (such as running GA/PSO followed by least_squares), and adding soft-constraints (such as ensure R1 < R2 or R1*C1 < 1 etc.). 

Aiming to create a consistent, [scikit-learn-like API](https://arxiv.org/abs/1309.0238) for impedance analysis, impedance.py contains modules for data preprocessing, validation, model fitting, and visualization.

For a little more in-depth discussion of the package background and capabilities, check out our [Journal of Open Source Software paper](https://joss.theoj.org/papers/10.21105/joss.02349).

If you have a feature request or find a bug, please [file an issue](https://github.com/k-vijayaraghavan/impedance_extend.py/issues) or, better yet, make the code improvements and [submit a pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)! The goal is to build an open-source tool that the entire impedance community can improve and use!

### Motivation for extension

The parameters of the equivalent circuit, $p$, need to be calculated such that the impedance of the circuit, $Z(f,p)$ is "equal" to the target impedance, $Z_{tgt}(f)$. Since the impedance cannot always be made exactly equal, it is instead more practical to minimize the difference between the impedances instead. To this end 
$$
p=\underset{p}{argmin} \; J(p)
$$
where the cost function, 
$$J(p)=\sum_{f} [Z(f,p)-Z_{tgt}(f)]^2$$

The original `impedance.py` uses `curve_fit` (which uses `least_squares` under the hood), and `basinhopping` when `global_opt=True`. `least_squares` is well suited for functions with a single minimum. If there are multiple minima, `least_squares` may get stuck at a local minimum; as such it is sensitive to initial conditions. `basinhopping` primarily uses local gradient methods to find the (local) minimum. The algorithm then uses a "hop" 
to perturb the parameter, and then checks if the cost function could have a logical global trend. 
Hence, the `basinhopping` works well if cost-function has "funnel" of "bowl" topology with some "pits".

We observed that `basinhopping` did not converge when data was noisy data. 
Hence, `impedance_extend` adds Particle Swarm Optimization (PSO) and Genetic Algorithm (GA) based optimization. 

PSO works when the cost-function land scape is "Rugged & Noisy" with non-differentiable mountain range with many small, disconnected "false valleys,".

GA works well when the cost-function landscape has local minimum with "Strong Local Attraction.". Since GA spans the parameter space it is likely to reach the true minimum. 

![optimization](docs/source/optimization.png)
**Illustration of the different optimization algorithms (AI generated).**

[Plevris et al.](https://joss.theoj.org/papers/10.21105/joss.02349) [10.21105/joss.02349](https://doi.org/10.21105/joss.02349) may provide better insights into the optimization problem (providing a head-to-head comparison between GA and PSO-based optimization and classical optimization for a wide rage of multi-parameter functions).

### Installation

The easiest way to install impedance_extend.py is from [PyPI](https://pypi.org/project/impedance_extend/) using pip.

```bash
pip install impedance_extend
```

See [Getting started with impedance.py](https://impedancepy.readthedocs.io/en/latest/getting-started.html) for instructions on getting started from scratch.

#### Dependencies

impedance.py requires:

-   Python (>=3.10)
-   SciPy (>=1.0)
-   NumPy (>=1.22.4)
-   Matplotlib (>=3.5)
-   Altair (>=3.0)
-   Pandas
-   Pygad (>=3.6.0)
-   Pyswarms (>=1.3)

Several example notebooks are provided in the `docs/source/examples/` directory. Opening these will require Jupyter notebook or Jupyter lab.

#### Examples and Documentation

Several examples can be found in the `docs/source/examples/` directory (the [Fitting impedance spectra notebook](https://impedancepy.readthedocs.io/en/latest/examples/fitting_example.html) is a great place to start) and the documentation can be found at [impedancepy.readthedocs.io](https://impedancepy.readthedocs.io/en/latest/).

## Citing the original impedance.py

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02349/status.svg)](https://doi.org/10.21105/joss.02349)

If you use impedance.py in published work, please consider citing https://joss.theoj.org/papers/10.21105/joss.02349 as

```bash
@article{Murbach2020,
  doi = {10.21105/joss.02349},
  url = {https://doi.org/10.21105/joss.02349},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2349},
  author = {Matthew D. Murbach and Brian Gerwe and Neal Dawson-Elli and Lok-kun Tsui},
  title = {impedance.py: A Python package for electrochemical impedance analysis},
  journal = {Journal of Open Source Software}
}
```

## Contributors ✨

This project was adapted from a fork of [impedance.py] (https://github.com/k-vijayaraghavan/impedance_extend.py). [Impedance.py] (https://github.com/k-vijayaraghavan/impedance_extend.py) was started at the [2018 Electrochemical Society (ECS) Hack Week in Seattle](https://www.electrochem.org/233/hack-week) and has benefited from a community of users and contributors since. Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/k-vijayaraghavan"><img src="https://avatars.githubusercontent.com/u/62916263?v=4" width="100px;" alt="Krishna Vijayaraghavan"/><br /><sub><b>Krishna Vijayaraghavan</b></sub></a><br /><a href="https://github.com/k-vijayaraghavan/impedance_extend.py/commits?author=k-vijayaraghavan" title="Code">💻</a></td>
    </tr>
    <tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
