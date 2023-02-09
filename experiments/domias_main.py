# future
from __future__ import absolute_import, division, print_function

# stdlib
import argparse
import os
from pathlib import Path
from typing import Dict, Union

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import multivariate_normal
from sdv.tabular import TVAE
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler
from synthcity.plugins import Plugins

# domias absolute
from domias.models.ctgan import CTGAN
from domias.models.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface

workspace = Path("synth_folder")
workspace.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gan_method",
    type=str,
    default="TVAE",
    choices=[
        "TVAE",
        "CTGAN",
        "KDE",
        "gaussian_copula",
        "adsgan",
        "tvae",
        "privbayes",
        "marginal_distributions",
        "bayesian_network",
        "ctgan",
        "copulagan",
        "nflow",
        "rtvae",
        "pategan",
    ],
    help="benchmarking generative model used for synthesis",
)
parser.add_argument(
    "--epsilon_adsgan", type=float, default=0.0, help="hyper-parameter in ads-gan"
)
parser.add_argument(
    "--density_estimator", type=str, default="prior", choices=["bnaf", "kde", "prior"]
)
parser.add_argument(
    "--training_size_list",
    nargs="+",
    type=int,
    default=[50],
    help="size of training dataset",
)
parser.add_argument(
    "--held_out_size_list",
    nargs="+",
    type=int,
    default=[1000],
    help="size of held-out dataset",
)
parser.add_argument(
    "--training_epoch_list",
    nargs="+",
    type=int,
    default=[2000],
    help="# training epochs",
)
parser.add_argument(
    "--gen_size_list",
    nargs="+",
    type=int,
    default=[10000],
    help="size of generated dataset",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu_idx", default=None, type=int)
parser.add_argument("--device", type=str, default=None)
parser.add_argument(
    "--dataset",
    type=str,
    default="SynthGaussian",
    choices=["housing", "synthetic", "Digits", "Covtype", "SynthGaussian"],
)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--batch_dim", type=int, default=50)
parser.add_argument("--clip_norm", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--cooldown", type=int, default=10)
parser.add_argument("--early_stopping", type=int, default=100)
parser.add_argument("--decay", type=float, default=0.5)
parser.add_argument("--min_lr", type=float, default=5e-4)
parser.add_argument("--polyak", type=float, default=0.998)
parser.add_argument("--flows", type=int, default=5)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument(
    "--residual", type=str, default="gated", choices=[None, "normal", "gated"]
)
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save", action="store_true")
parser.add_argument("--tensorboard", type=str, default="tensorboard")
parser.add_argument("--shifted_column", type=int, default=None)
parser.add_argument("--zero_quantile", type=float, default=0.3)
parser.add_argument("--reference_kept_p", type=float, default=1.0)

args = parser.parse_args()
args.device = DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alias = f"v3kde1_shift{args.shifted_column}_zq{args.zero_quantile}_kp{args.reference_kept_p}_{args.batch_dim}_{args.hidden_dim}_{args.layers}_{args.epochs}_{args.gan_method}_{args.epsilon_adsgan}_{args.density_estimator}_{args.dataset}_trn_sz{args.training_size_list}_ref_sz{args.held_out_size_list}_gen_sz{args.gen_size_list}_{args.seed}"

if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results_folder", exist_ok=True)

performance_logger: Dict = {}

if args.dataset == "housing":

    def data_loader() -> np.ndarray:
        # np.random.multivariate_normal([0],[[1]], n1)*std1 # non-training data
        scaler = StandardScaler()
        X = fetch_california_housing().data
        np.random.shuffle(X)
        return scaler.fit_transform(X)

    dataset = data_loader()
    print("dataset size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "synthetic":
    dataset = np.load(f"../dataset/synthetic_gaussian_{20}_{10}_{30000}_train.npy")
    print("dataset size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "Digits":
    scaler = StandardScaler()
    dataset = load_digits().data
    dataset = scaler.fit_transform(dataset)
    np.random.seed(1)
    np.random.shuffle(dataset)
    # X, y = dataset['data'], dataset['target']
    print("dataset size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "Covtype":
    scaler = StandardScaler()
    dataset = fetch_covtype().data
    dataset = scaler.fit_transform(dataset)
    np.random.seed(1)
    np.random.shuffle(dataset)
    # X, y = dataset['data'], dataset['target']-1
    print("dataset size,", dataset.shape)
    ndata = dataset.shape[0]

elif args.dataset == "SynthGaussian":
    dataset = np.random.randn(20000, 3)
    ndata = dataset.shape[0]


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
    epsilon_adsgan: float = 0,
    seed: int = 0,
) -> GeneratorInterface:
    class LocalGenerator(GeneratorInterface):
        def __init__(self) -> None:
            if gan_method == "TVAE":
                syn_model = TVAE(epochs=TRAINING_EPOCH)
            elif gan_method == "CTGAN":
                syn_model = CTGAN(epochs=TRAINING_EPOCH)
            elif gan_method == "KDE":
                syn_model = None
            else:  # synthcity
                syn_model = Plugins().get(gan_method)
                if gan_method == "adsgan":
                    syn_model.lambda_identifiability_penalty = epsilon_adsgan
                    syn_model.seed = seed
                elif gan_method == "pategan":
                    syn_model.dp_delta = 1e-5
                    syn_model.dp_epsilon = epsilon_adsgan
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


""" 2. training-test-addition split"""
for SIZE_PARAM in args.training_size_list:
    for ADDITION_SIZE in args.held_out_size_list:
        for TRAINING_EPOCH in args.training_epoch_list:
            if SIZE_PARAM * 2 + ADDITION_SIZE >= ndata:
                continue
            """
            Process the dataset for covariant shift experiments
            """

            generator = get_generator(
                gan_method=args.gan_method,
                epsilon_adsgan=args.epsilon_adsgan,
                seed=args.gpu_idx if args.gpu_idx is not None else 0,
            )
            evaluate_performance(
                generator,
                dataset,
                SIZE_PARAM,
                ADDITION_SIZE,
                TRAINING_EPOCH,
                shifted_column=args.shifted_column,
                zero_quantile=args.zero_quantile,
                seed=args.gpu_idx if args.gpu_idx is not None else 0,
                density_estimator=args.density_estimator,
                reference_kept_p=args.reference_kept_p,
                gen_size_list=args.gen_size_list,
                workspace=workspace,
                device=DEVICE,
            )
