# future
from __future__ import absolute_import, division, print_function

# stdlib
import argparse
import os
from typing import Dict

# third party
import numpy as np
import pandas as pd
import torch
from ctgan import CTGANSynthesizer
from scipy import stats
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.metrics.combined import compute_metrics

PATH_CELEB_REPRESENTATION = "celebA_representation"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gan_method",
    type=str,
    default="adsgan",
    choices=[
        "TVAE",
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
    "--epsilon_adsgan", type=float, default=0.1, help="hyper-parameter in ads-gan"
)
parser.add_argument(
    "--density_estimator", type=str, default="bnaf", choices=["bnaf", "kde"]
)
parser.add_argument(
    "--training_size_list",
    nargs="+",
    type=int,
    default=[999],
    help="size of training dataset",
)
parser.add_argument(
    "--held_out_size_list",
    nargs="+",
    type=int,
    default=[4500],
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
    default=[50000],
    help="size of generated dataset",
)
parser.add_argument("--device", type=str, default=None)
parser.add_argument(
    "--dataset",
    type=str,
    default="CelebA",
    choices=["housing", "synthetic", "CelebA"],
)

parser.add_argument("--gpu_idx", default=2, type=int)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--rep_dim", type=int, default=128)
parser.add_argument("--ae_epoch", type=int, default=100)
parser.add_argument("--dcgan_epoch", type=int, default=1000)
parser.add_argument("--training_size", type=int, default=1000)


args = parser.parse_args()
args.device = f"cuda:{args.gpu_idx}"


alias = f"{args.seed}_{args.gpu_idx}_{args.rep_dim}_tsz{args.training_size}_gsz{args.gen_size_list[0]*2}"


if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results_folder", exist_ok=True)


performance_logger: Dict = {}

""" 1. load dataset"""
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
    dataset = np.load(
        f"{PATH_CELEB_REPRESENTATION}/dataset/synthetic_gaussian_{20}_{10}_{30000}_train.npy"
    )
    print("dataset size,", dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "CelebA":
    training_set = np.load(
        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_real_{alias}.npy"
    )
    test_set = np.load(
        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_test_{alias}.npy"
    )[: 999 + args.training_size - 1000]
    addition_set = np.load(
        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_ref_{alias}.npy"
    )[:4500]
    addition_set2 = np.load(
        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_ref_{alias}.npy"
    )[4500:]

""" 2. training-test-addition split"""
for SIZE_PARAM in args.training_size_list:
    for ADDITION_SIZE in args.held_out_size_list:
        for TRAINING_EPOCH in args.training_epoch_list:
            performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"] = {}
            if args.dataset != "CelebA":
                pass
            else:
                for N_DATA_GEN in args.gen_size_list:
                    samples = pd.DataFrame(
                        np.load(
                            f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_synth_{alias}.npy"
                        )[:N_DATA_GEN]
                    )
                    samples_val = pd.DataFrame(
                        np.load(
                            f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_synth_{alias}.npy"
                        )[N_DATA_GEN : N_DATA_GEN * 2]
                    )
                # eval_met = evaluate(pd.DataFrame(samples), df)
                # eval_met = compute_metrics(samples, dataset[:N_DATA_GEN], which_metric = ['WD'])['wd_measure']
                wd_n = min(len(samples), len(addition_set))
                eval_met_on_held_out = compute_metrics(
                    samples[:wd_n], addition_set[:wd_n], which_metric=["WD"]
                )["wd_measure"]
                # eval_ctgan = evaluate(samples, pd.DataFrame(addition_set2))
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_evaluation"
                ] = eval_met_on_held_out
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
                    eval_met_on_held_out,
                )

                """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
                if args.density_estimator == "bnaf":
                    _gen, model_gen = density_estimator_trainer(
                        samples.values,
                        samples_val.values[: int(0.5 * N_DATA_GEN)],
                        samples_val.values[int(0.5 * N_DATA_GEN) :],
                    )
                    _data, model_data = density_estimator_trainer(
                        addition_set,
                        addition_set2[: int(0.5 * ADDITION_SIZE)],
                        addition_set2[: int(0.5 * ADDITION_SIZE)],
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
                    density_data = stats.gaussian_kde(addition_set.transpose(1, 0))
                    p_G_train = density_gen(training_set.transpose(1, 0))
                    p_G_test = density_gen(test_set.transpose(1, 0))

                X_test_4baseline = np.concatenate([training_set, test_set])
                Y_test_4baseline = np.concatenate(
                    [np.ones(training_set.shape[0]), np.zeros(test_set.shape[0])]
                ).astype(bool)
                # build another GAN for hayes and GAN_leak_cal
                ctgan = CTGANSynthesizer(epochs=200)
                samples.columns = [str(_) for _ in range(training_set.shape[1])]
                ctgan.fit(samples)  # train a CTGAN on the generated examples

                ctgan_representation = ctgan._transformer.transform(X_test_4baseline)
                print(ctgan_representation.shape)
                ctgan_score = (
                    ctgan._discriminator(
                        torch.as_tensor(ctgan_representation).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                print(ctgan_score.shape)

                acc, auc = compute_metrics_baseline(ctgan_score, Y_test_4baseline)

                X_ref_GLC = ctgan.sample(addition_set.shape[0])

                baseline_results, baseline_scores = baselines(
                    X_test_4baseline,
                    Y_test_4baseline,
                    samples.values,
                    addition_set,
                    X_ref_GLC,
                )
                baseline_results = baseline_results.append(
                    {"name": "hayes", "acc": acc, "auc": auc}, ignore_index=True
                )
                baseline_scores["hayes"] = ctgan_score
                print("baselines:", baseline_results)
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Baselines"
                ] = baseline_results
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_BaselineScore"
                ] = baseline_scores
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Xtest"
                ] = X_test_4baseline
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Ytest"
                ] = Y_test_4baseline

                #                 eqn1: \prop P_G(x_i)
                log_p_test = np.concatenate([p_G_train, p_G_test])
                thres = np.quantile(log_p_test, 0.5)
                auc_y = np.hstack(
                    (
                        np.array([1] * training_set.shape[0]),
                        np.array([0] * test_set.shape[0]),
                    )
                )
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_test, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                print(
                    "Eqn.(1), training set prediction acc",
                    (p_G_train > thres).sum(0) / SIZE_PARAM,
                )
                print("Eqn.(1), AUC", auc)
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn1"
                ] = (p_G_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn1AUC"
                ] = auc
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn1Score"
                ] = log_p_test
                # eqn2: \prop P_G(x_i)/P_X(x_i)
                if args.density_estimator == "bnaf":
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
                elif args.density_estimator == "kde":
                    p_R_train = density_data(training_set.transpose(1, 0)) + 1e-30
                    p_R_test = density_data(test_set.transpose(1, 0)) + 1e-30

                if args.density_estimator == "bnaf":
                    log_p_rel = np.concatenate(
                        [p_G_train - p_R_train, p_G_test - p_R_test]
                    )
                elif args.density_estimator == "kde":
                    log_p_rel = np.concatenate(
                        [p_G_train / p_R_train, p_G_test / p_R_test]
                    )

                thres = np.quantile(log_p_rel, 0.5)
                auc_y = np.hstack(
                    (
                        np.array([1] * training_set.shape[0]),
                        np.array([0] * test_set.shape[0]),
                    )
                )
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_rel, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                if args.density_estimator == "bnaf":
                    print(
                        "Eqn.(2), training set prediction acc",
                        (p_G_train - p_R_train >= thres).sum(0) / SIZE_PARAM,
                    )
                    print("Eqn.(2), AUC", auc)
                    performance_logger[
                        f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"
                    ][f"{N_DATA_GEN}_Eqn2"] = (p_G_train - p_R_train > thres).sum(
                        0
                    ) / SIZE_PARAM
                elif args.density_estimator == "kde":
                    print(
                        "Eqn.(2), training set prediction acc",
                        (p_G_train / p_R_train >= thres).sum(0) / SIZE_PARAM,
                    )
                    print("Eqn.(2), AUC", auc)
                    performance_logger[
                        f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"
                    ][f"{N_DATA_GEN}_Eqn2"] = (p_G_train / p_R_train > thres).sum(
                        0
                    ) / SIZE_PARAM
                # print('Eqn.(2), test set prediction acc', (p_G_test-p_R_test > thres).sum(0) / SIZE_PARAM)

                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn2AUC"
                ] = auc
                performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                    f"{N_DATA_GEN}_Eqn2Score"
                ] = log_p_rel

                if args.gan_method == "adsgan":
                    np.save(
                        f"results_folder/AISTATS_CELEBA_{alias}_gsz{args.gen_size_list}.npy",
                        performance_logger,
                    )
                elif args.gan_method == "pategan":
                    np.save(
                        f"results_folder/AISTATS_CELEBA_{alias}_gsz{args.gen_size_list}.npy",
                        performance_logger,
                    )
                else:
                    np.save(
                        f"results_folder/AISTATS_CELEBA_{alias}_gsz{args.gen_size_list}.npy",
                        performance_logger,
                    )
