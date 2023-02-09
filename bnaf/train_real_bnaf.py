import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gan_method",
    type=str,
    default="TVAE",
    help="benchmarking generative model used for synthesis",
)
parser.add_argument(
    "--density_estimator", type=str, default="bnaf", choices=["bnaf", "kde"]
)
parser.add_argument(
    "--training_size_list",
    nargs="+",
    type=int,
    default=[50, 500, 1000, 5000, 10000],
    help="size of training dataset",
)
parser.add_argument(
    "--held_out_size_list",
    nargs="+",
    type=int,
    default=[10000],
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
    default=[100, 500, 1000, 5000, 10000],
    help="size of generated dataset",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu_idx", default=2, type=int)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument(
    "--dataset",
    type=str,
    default="housing",
    choices=["MAGGIC", "housing", "synthetic"],
)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--batch_dim", type=int, default=200)
parser.add_argument("--clip_norm", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--cooldown", type=int, default=10)
parser.add_argument("--early_stopping", type=int, default=100)
parser.add_argument("--decay", type=float, default=0.5)
parser.add_argument("--min_lr", type=float, default=5e-4)
parser.add_argument("--polyak", type=float, default=0.998)
parser.add_argument("--flows", type=int, default=5)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=10)
parser.add_argument(
    "--residual", type=str, default="gated", choices=[None, "normal", "gated"]
)
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save", action="store_false")
parser.add_argument("--tensorboard", type=str, default="tensorboard")

args = parser.parse_args()
args.device = f"cuda:{args.gpu_idx}"

if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from bnaf_den_est import *
import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer
from ctgan import load_demo
import ctgan
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
from scipy import stats
from sklearn import metrics
from metrics.combined import compute_metrics
import os

os.makedirs("results_folder", exist_ok=True)
performance_logger = {}

""" 1. load dataset"""
if args.dataset == "MAGGIC":
    dataset = getattr(datasets, "MAGGIC")("")
    dataset = np.vstack((dataset.train.x, dataset.val.x, dataset.test.x))
    print("training size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "housing":
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    def data_loader():
        # np.random.multivariate_normal([0],[[1]], n1)*std1 # non-training data
        scaler = StandardScaler()
        X = fetch_california_housing().data
        np.random.shuffle(X)
        return scaler.fit_transform(X)

    dataset = data_loader()
    print("training size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "synthetic":
    dataset = np.load(f"../dataset/synthetic_gaussian_{20}_{10}_{30000}_train.npy")
    print("training size,", dataset.shape)
    ndata = dataset.shape[0]


""" 2. training-test-addition split"""
for SIZE_PARAM in args.training_size_list:
    for ADDITION_SIZE in args.held_out_size_list:
        for TRAINING_EPOCH in args.training_epoch_list:
            if SIZE_PARAM * 2 + ADDITION_SIZE >= ndata:
                continue
            performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"] = {}
            training_set = dataset[:SIZE_PARAM]
            test_set = dataset[SIZE_PARAM : 2 * SIZE_PARAM]
            addtion_set = dataset[-ADDITION_SIZE:]
            addtion_set2 = dataset[-2 * ADDITION_SIZE : -ADDITION_SIZE]
            """ 3. Synthesis with TVAE"""
            df = pd.DataFrame(training_set)
            df.columns = [str(_) for _ in range(dataset.shape[1])]
            if args.gan_method == "TVAE":
                tvae_model = TVAE(epochs=TRAINING_EPOCH)
                tvae_model.fit(df)
            elif args.gan_method == "CTGAN":
                tvae_model = CTGANSynthesizer(epochs=TRAINING_EPOCH)
                tvae_model.fit(df)
            for N_DATA_GEN in args.gen_size_list:
                samples = tvae_model.sample(N_DATA_GEN)
                samples_val = tvae_model.sample(N_DATA_GEN)
                # eval_met = evaluate(pd.DataFrame(samples), df)
                eval_met = compute_metrics(
                    samples, dataset[:N_DATA_GEN], which_metric=["WD"]
                )["wd_measure"]
                eval_met_on_held_out = compute_metrics(
                    samples, addtion_set[:N_DATA_GEN], which_metric=["WD"]
                )["wd_measure"]
                eval_ctgan = evaluate(pd.DataFrame(samples), df)
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_evaluation"
                ] = (eval_met, eval_ctgan, eval_met_on_held_out)
                print(
                    "SIZE: ",
                    SIZE_PARAM,
                    "TVAE EPOCH: ",
                    TRAINING_EPOCH,
                    "N_DATA_GEN: ",
                    N_DATA_GEN,
                    "ADDITION_SIZE: ",
                    ADDITION_SIZE,
                    "Performance (Sample-Quality): ",
                    eval_met,
                    eval_ctgan,
                )

                """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
                if args.density_estimator == "bnaf":
                    _gen, model_gen = density_estimator_trainer(
                        samples.values,
                        samples_val.values[: int(0.5 * N_DATA_GEN)],
                        samples_val.values[int(0.5 * N_DATA_GEN) :],
                        args=args,
                    )
                    _data, model_data = density_estimator_trainer(
                        addtion_set,
                        addtion_set2[: int(0.5 * ADDITION_SIZE)],
                        addtion_set2[: int(0.5 * ADDITION_SIZE)],
                        args=args,
                    )
                    p_G_train = (
                        compute_log_p_x(
                            model_gen, torch.as_tensor(training_set).float().to(device)
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    p_G_test = (
                        compute_log_p_x(
                            model_gen, torch.as_tensor(test_set).float().to(device)
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                elif args.density_estimator == "kde":
                    density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
                    density_data = stats.gaussian_kde(addtion_set.transpose(1, 0))
                    p_G_train = density_gen(training_set.transpose(1, 0))
                    p_G_test = density_gen(test_set.transpose(1, 0))
                # eqn1: \prop P_G(x_i)

                thres = np.quantile(np.vstack((p_G_train, p_G_test)), 0.5)

                auc_y = np.hstack(
                    (
                        np.array([1] * training_set.shape[0]),
                        np.array([0] * test_set.shape[0]),
                    )
                )
                auc_pred = np.hstack((p_G_train, p_G_test))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, auc_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                print(
                    "Eqn.(1), training set prediction acc",
                    (p_G_train > thres).sum(0) / SIZE_PARAM,
                )
                print(
                    "Eqn.(1), test set prediction acc",
                    (p_G_test > thres).sum(0) / SIZE_PARAM,
                )
                print("Eqn.(1), AUC", auc)
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn1"
                ] = (p_G_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn1AUC"
                ] = auc

                # eqn2: \prop P_G(x_i)/P_X(x_i)
                p_R_train = (
                    compute_log_p_x(
                        model_data, torch.as_tensor(training_set).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                p_R_test = (
                    compute_log_p_x(
                        model_data, torch.as_tensor(test_set).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                #                 p_R_train = density_data(training_set.transpose(1,0))
                #                 p_R_test = density_data(test_set.transpose(1,0))

                thres = np.quantile(
                    np.vstack((p_G_train - p_R_train, p_G_test - p_R_test)), 0.5
                )

                auc_y = np.hstack(
                    (
                        np.array([1] * training_set.shape[0]),
                        np.array([0] * test_set.shape[0]),
                    )
                )
                auc_pred = np.hstack((p_G_train - p_R_train, p_G_test - p_R_test))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, auc_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                print(
                    "Eqn.(2), training set prediction acc",
                    (p_G_train - p_R_train > thres).sum(0) / SIZE_PARAM,
                )
                print(
                    "Eqn.(2), test set prediction acc",
                    (p_G_test - p_R_test > thres).sum(0) / SIZE_PARAM,
                )
                print("Eqn.(2), AUC", auc)
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn2"
                ] = (p_G_train / p_R_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn2AUC"
                ] = auc
                np.save(
                    f"results_folder/performance_logger_{args.gan_method}_{args.seed}_{args.dataset}.npy",
                    performance_logger,
                )
