# future
from __future__ import absolute_import, division, print_function

# stdlib
from typing import Union

# third party
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import multivariate_normal
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
            # np.random.multivariate_normal([0],[[1]], n1)*std1 # non-training data
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


class gaussian:
    def __init__(self, X: np.ndarray) -> None:
        var = np.std(X, axis=0) ** 2
        mean = np.mean(X, axis=0)
        self.rv = multivariate_normal(mean, np.diag(var))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return self.rv.pdf(Z)


class normal_func:
    def __init__(self, X: np.ndarray) -> None:
        self.var = np.ones_like(np.std(X, axis=0) ** 2)
        self.mean = np.zeros_like(np.mean(X, axis=0))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z, self.mean, np.diag(self.var))
        # return multivariate_normal.pdf(Z, np.zeros_like(self.mean), np.diag(np.ones_like(self.var)))


class normal_func_feat:
    def __init__(
        self,
        X: np.ndarray,
        continuous: Union[list, str, np.ndarray] = [1, 0, 0, 0, 0, 0, 0, 0],
    ) -> None:
        if continuous == "all":
            self.feat = np.ones(X.shape[1]).astype(bool)
        else:
            if np.any(np.array(continuous) > 1) or len(continuous) != X.shape[1]:
                raise ValueError("Continous variable needs to be boolean")
            self.feat = np.array(continuous).astype(bool)

        if np.sum(self.feat) == 0:
            raise ValueError("there needs to be at least one continuous feature")

        for i in np.arange(X.shape[1])[self.feat]:
            if len(np.unique(X[:, i])) < 10:
                print(f"Warning: feature {i} does not seem continous. CHECK")

        self.var = np.std(X[:, self.feat], axis=0) ** 2
        self.mean = np.mean(X[:, self.feat], axis=0)
        # self.rv = multivariate_normal(mean, np.diag(var))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z[:, self.feat], self.mean, np.diag(self.var))


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
            if self.method == "KDE":
                self.model = stats.gaussian_kde(data.transpose(1, 0))
            else:
                self.model.fit(data)

            return self

        def generate(self, count: int) -> pd.DataFrame:
            if gan_method == "KDE":
                samples = pd.DataFrame(self.model.resample(count).transpose(1, 0))
            elif gan_method == "TVAE":
                samples = self.model.sample(count)
            else:  # synthcity
                samples = self.model.generate(count=count)

            return samples

    return LocalGenerator()


@pytest.mark.parametrize("dataset_name", ["housing", "Covtype", "SynthGaussian"])
@pytest.mark.parametrize("method", ["TVAE", "CTGAN", "KDE"])
@pytest.mark.parametrize("training_size", [30])
@pytest.mark.parametrize("held_out_size", [30])
@pytest.mark.parametrize("training_epoch", [100])
def test_sanity(
    dataset_name: str,
    method: str,
    training_size: int,
    held_out_size: int,
    training_epoch: int,
) -> None:
    dataset = get_dataset(dataset_name)
    print(dataset)

    generator = get_generator(
        gan_method=method,
        epochs=training_epoch,
    )
    perf = evaluate_performance(
        generator,
        dataset,
        training_size,
        held_out_size,
        training_epoch,
        gen_size_list=[100],
    )
    print(
        f"""
            SIZE_PARAM = {training_size} ADDITION_SIZE  = {held_out_size} TRAINING_EPOCH = {training_epoch}
                metrics = {perf}
        """
    )
