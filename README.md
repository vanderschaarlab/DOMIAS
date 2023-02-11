# DOMIAS: Membership Inference Attacks against Synthetic Data through Overfitting Detection

<div align="center">

[![Tests Python](https://github.com/vanderschaarlab/DOMIAS/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/DOMIAS/actions/workflows/test.yml)
[![](https://pepy.tech/badge/DOMIAS)](https://pypi.org/project/synthcity/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/DOMIAS/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)

</div>

## :rocket: Installation

The library can be installed from PyPI using
```bash
$ pip install domias
```
or from source, using
```bash
$ pip install .
```

## Experiments

1. **Experiments main paper**

To reproduce results for DOMIAS, baselines, and ablated models, run
```python
cd experiments
python3 domias_main.py --seed 0 --gan_method TVAE --dataset housing --training_size_list 30 50 100 300 500 1000 --held_out_size_list 10000 --gen_size_list 10000 --training_epoch_list 2000
```
changing arguments training_size_list, held_out_size_list, gen_size_list, and training_epoch_list for specific experiments over ranges (Experiments 5.1 and 5.2, see Appendix A for details) and gan_method for generative model of interest.

or equivalently, run
```python
cd experiments && bash run_tabular.sh
```

2. **Experiments no prior knowledge (Appendix D)**

If using prior knowledge (i.e., no reference dataset setting), add
```python
--density_estimator prior
```

3. **Experiment images (Appendix B.3)**

__Note__: The CelebA dataset must be available in the `experiments/data` folder.

To run experiment with the CelebA dataset, first run
```python
cd experiments && python3 celeba_gen.py --seed 0 --training_size 4000
```
and then
```python
cd experiments && python3 celeba_eval.py --seed 0 --training_size 4000
```
## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```
## Citing

TODO
