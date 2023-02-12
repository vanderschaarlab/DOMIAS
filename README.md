# DOMIAS: Membership Inference Attacks against Synthetic Data through Overfitting Detection

<div align="center">

[![Tests Python](https://github.com/vanderschaarlab/DOMIAS/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/DOMIAS/actions/workflows/test.yml)
[![](https://pepy.tech/badge/domias)](https://pypi.org/project/domias/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vanderschaarlab/DOMIAS/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)

</div>

## Installation

The library can be installed from PyPI using
```bash
$ pip install domias
```
or from source, using
```bash
$ pip install .
```

## API
The main API call is
```python
from domias.evaluator import evaluate_performance
```

`evaluate_performance` expects as input a generator which implements the `domias.models.generator.GeneratorInterface` interface, and an evaluation dataset.

The supported arguments for `evaluate_performance` are:
```
  generator: GeneratorInterface
      Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
  dataset: int
      The evaluation dataset, used to derive the training and test datasets.
  training_size: int
      The split for the training dataset out of `dataset`
  held_out_size: int
      The split for the held-out(addition) dataset out of `dataset`.
  training_epochs: int
      Training epochs
  synthetic_sizes: List[int]
      For how many synthetic samples to test the attacks.
  density_estimator: str, default = "prior"
      Which density to use. Available options:
          * prior
          * bnaf
          * kde
  seed: int
      Random seed
  device: PyTorch device
      CPU or CUDA
  shifted_column: Optional[int]
      Shift a column
  zero_quantile: float
      Threshold for shifting the column.
  reference_kept_p: float
      Held-out dataset parameter
```

The output consists of dictionary with a key for each of the `synthetic_sizes` values.

For each `synthetic_sizes` value, the dictionary contains the keys:
 - `MIA_performance` : accuracy and AUCROC for each attack
 - `MIA_scores`: output scores for each attack
 - `data`: the evaluation data

 For both `MIA_performance` and `MIA_scores`, the following attacks are evaluated:
 - "baseline_eq1"
 - "baseline_eq2"
 - "hayes_torch"
 - "hilprecht"
 - "gan_leaks"
 - "gan_leaks_cal"
 - "hayes_gan"
 - "eq1"
 - "domias"

## Sample usage

Example for using `evaluate_performance`:
```python
# third party
import pandas as pd
from sdv.tabular import TVAE

# domias absolute
from domias.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface


def get_generator(
    epochs: int = 1000,
    seed: int = 0,
) -> GeneratorInterface:
    class LocalGenerator(GeneratorInterface):
        def __init__(self) -> None:
            self.model = TVAE(epochs=epochs)

        def fit(self, data: pd.DataFrame) -> "LocalGenerator":
            self.model.fit(data)
            return self

        def generate(self, count: int) -> pd.DataFrame:
            return self.model.sample(count)

    return LocalGenerator()


dataset = ...  # Load your dataset as numpy array

training_size = 1000
held_out_size = 1000
training_epochs = 2000
synthetic_sizes = [1000]
density_estimator = "prior"  # prior, kde, bnaf

generator = get_generator(
    epochs=training_epochs,
)

perf = evaluate_performance(
    generator,
    dataset,
    training_size,
    held_out_size,
    training_epochs=training_epochs,
    synthetic_sizes=[100],
    density_estimator=density_estimator,
)

assert 100 in perf
results = perf[100]

assert "MIA_performance" in results
assert "MIA_scores" in results

print(results["MIA_performance"])
```

## Experiments

1. **Experiments main paper**

To reproduce results for DOMIAS, baselines, and ablated models, run
```bash
cd experiments
python3 domias_main.py --seed 0 --gan_method TVAE --dataset housing --training_size_list 30 50 100 300 500 1000 --held_out_size_list 10000 --synthetic_sizes 10000 --training_epoch_list 2000
```
changing arguments training_size_list, held_out_size_list, synthetic_sizes, and training_epoch_list for specific experiments over ranges (Experiments 5.1 and 5.2, see Appendix A for details) and gan_method for generative model of interest.

or equivalently, run
```bash
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
```bash
cd experiments && python3 celeba_gen.py --seed 0 --training_size 4000
```
and then
```bash
cd experiments && python3 celeba_eval.py --seed 0 --training_size 4000
```
## Tests

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
