# future
from __future__ import absolute_import, division, print_function

# stdlib
import argparse
import os
from pathlib import Path

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sdv.tabular import TVAE
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_digits
from sklearn.preprocessing import StandardScaler
from synthcity.plugins import Plugins

# domias absolute
from domias.evaluator import evaluate_performance
from domias.models.ctgan import CTGAN
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
parser.add_argument("--shifted_column", type=int, default=None)
parser.add_argument("--zero_quantile", type=float, default=0.3)
parser.add_argument("--reference_kept_p", type=float, default=1.0)

args = parser.parse_args()
args.device = DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results_folder", exist_ok=True)

if args.dataset == "housing":

    def data_loader() -> np.ndarray:
        # np.random.multivariate_normal([0],[[1]], n1)*std1 # non-training data
        scaler = StandardScaler()
        X = fetch_california_housing().data
        np.random.shuffle(X)
        return scaler.fit_transform(X)

    dataset = data_loader()
    ndata = dataset.shape[0]
elif args.dataset == "Digits":
    scaler = StandardScaler()
    dataset = load_digits().data
    dataset = scaler.fit_transform(dataset)
    np.random.seed(1)
    np.random.shuffle(dataset)
    ndata = dataset.shape[0]
elif args.dataset == "Covtype":
    scaler = StandardScaler()
    dataset = fetch_covtype().data
    dataset = scaler.fit_transform(dataset)
    np.random.seed(1)
    np.random.shuffle(dataset)
    ndata = dataset.shape[0]

elif args.dataset == "SynthGaussian":
    dataset = np.random.randn(20000, 3)
    ndata = dataset.shape[0]


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
                self.model = stats.gaussian_kde(np.transpose(data))
            else:
                self.model.fit(data)

            return self

        def generate(self, count: int) -> pd.DataFrame:
            if gan_method == "KDE":
                samples = pd.DataFrame(np.transpose(self.model.resample(count)))
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
            perf = evaluate_performance(
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
            print(
                f"""
                SIZE_PARAM = {SIZE_PARAM} ADDITION_SIZE  = {ADDITION_SIZE} TRAINING_EPOCH = {TRAINING_EPOCH}
                    metrics = {perf}
            """
            )
