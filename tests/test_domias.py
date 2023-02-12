# future
from __future__ import absolute_import, division, print_function

# third party
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sdv.tabular import TVAE
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler

# domias absolute
from domias.evaluator import evaluate_performance
from domias.models.ctgan import CTGAN
from domias.models.generator import GeneratorInterface


def get_dataset(dataset: str) -> np.ndarray:
    if dataset == "housing":

        def data_loader() -> np.ndarray:
            scaler = StandardScaler()
            X = fetch_california_housing().data
            np.random.shuffle(X)
            return scaler.fit_transform(X)

        dataset = data_loader()
    elif dataset == "Digits":
        scaler = StandardScaler()
        dataset = load_digits().data
        dataset = scaler.fit_transform(dataset)
        np.random.seed(1)
        np.random.shuffle(dataset)
    elif dataset == "Covtype":
        scaler = StandardScaler()
        dataset = fetch_covtype().data
        dataset = scaler.fit_transform(dataset)
        np.random.seed(1)
        np.random.shuffle(dataset)
    elif dataset == "SynthGaussian":
        dataset = np.random.randn(20000, 3)

    return dataset[:1000]


def get_generator(
    gan_method: str = "TVAE",
    epochs: int = 100,
    seed: int = 0,
) -> GeneratorInterface:
    class LocalGenerator(GeneratorInterface):
        def __init__(self) -> None:
            if gan_method == "TVAE":
                syn_model = TVAE(epochs=epochs)
            elif gan_method == "CTGAN":
                syn_model = CTGAN(epochs=epochs)
            elif gan_method == "KDE":
                syn_model = None
            else:
                raise RuntimeError()
            self.method = gan_method
            self.model = syn_model

        def fit(self, data: pd.DataFrame) -> "LocalGenerator":
            self.lower = data.min().min()
            self.upper = data.max().max()

            if self.method == "KDE":
                self.model = stats.gaussian_kde(np.transpose(data))
            else:
                self.model.fit(data)

            return self

        def generate(self, count: int) -> pd.DataFrame:
            if gan_method == "KDE":
                samples = pd.DataFrame(self.model.resample(count).transpose(1, 0))
            elif gan_method == "TVAE":
                samples = self.model.sample(count)
            elif gan_method == "CTGAN":
                samples = self.model.generate(count)
            else:
                raise RuntimeError()

            return samples

    return LocalGenerator()


@pytest.mark.parametrize("dataset_name", ["housing", "SynthGaussian"])
@pytest.mark.parametrize("method", ["TVAE", "CTGAN", "KDE"])
@pytest.mark.parametrize("training_size", [30])
@pytest.mark.parametrize("held_out_size", [30])
@pytest.mark.parametrize("training_epochs", [100])
@pytest.mark.parametrize("density_estimator", ["prior", "kde", "bnaf"])
def test_sanity(
    dataset_name: str,
    method: str,
    training_size: int,
    held_out_size: int,
    training_epochs: int,
    density_estimator: str,
) -> None:
    dataset = get_dataset(dataset_name)

    generator = get_generator(
        gan_method=method,
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

    tests = [
        "baseline_eq1",
        "baseline_eq2",
        "hayes_torch",
        "hilprecht",
        "gan_leaks",
        "gan_leaks_cal",
        "hayes_gan",
        "eq1",
        "domias",
    ]

    for key in tests:
        assert key in results["MIA_performance"]
        assert key in results["MIA_scores"]
        assert "accuracy" in results["MIA_performance"][key]
        assert "aucroc" in results["MIA_performance"][key]
