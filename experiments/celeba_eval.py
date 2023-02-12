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
from scipy import stats
from sklearn import metrics

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.metrics.wd import compute_wd
from domias.models.ctgan import CTGAN

PATH_CELEB_REPRESENTATION = "celebA_representation"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--density_estimator", type=str, default="bnaf", choices=["bnaf", "kde"]
)
parser.add_argument(
    "--training_size_list",
    nargs="+",
    type=int,
    default=[100],  # 999
    help="size of training dataset",
)
parser.add_argument(
    "--held_out_size_list",
    nargs="+",
    type=int,
    default=[100],  # 4500
    help="size of held-out dataset",
)
parser.add_argument(
    "--training_epoch_list",
    nargs="+",
    type=int,
    default=[100],  # 2000
    help="# training epochs",
)
parser.add_argument(
    "--synthetic_sizes",
    nargs="+",
    type=int,
    default=[100],  # 50000
    help="size of generated dataset",
)
parser.add_argument("--device", type=str, default=None)

parser.add_argument("--gpu_idx", default=None, type=int)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--rep_dim", type=int, default=128)
parser.add_argument("--training_size", type=int, default=4000)


args = parser.parse_args()
args.device = f"cuda:{args.gpu_idx}"

GPU_IDX = args.gpu_idx
SEED = args.seed
LATENT_REPRESENTATION_DIM = args.rep_dim

alias = f"{SEED}_{GPU_IDX}_{LATENT_REPRESENTATION_DIM}_tsz{args.training_size}"


if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results_folder", exist_ok=True)


performance_logger: Dict = {}

""" 1. load dataset"""
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
for training_size in args.training_size_list:
    for held_out_size in args.held_out_size_list:
        for training_epochs in args.training_epoch_list:
            performance_logger[
                f"{training_size}_{training_epochs}_{held_out_size}"
            ] = {}
            for N_DATA_GEN in args.synthetic_sizes:
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
            wd_n = min(len(samples), len(addition_set))
            eval_met_on_held_out = compute_wd(samples[:wd_n], addition_set[:wd_n])
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_evaluation"
            ] = eval_met_on_held_out
            print(
                "SIZE: ",
                training_size,
                "TVAE EPOCH: ",
                training_epochs,
                "N_DATA_GEN: ",
                N_DATA_GEN,
                "held_out_size: ",
                held_out_size,
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
                    addition_set2[: int(0.5 * held_out_size)],
                    addition_set2[: int(0.5 * held_out_size)],
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
            ctgan = CTGAN(epochs=200)
            samples.columns = [str(_) for _ in range(training_set.shape[1])]
            ctgan.fit(samples)  # train a CTGAN on the generated examples

            if ctgan._transformer is None or ctgan._discriminator is None:
                raise RuntimeError()

            ctgan_representation = ctgan._transformer.transform(X_test_4baseline)
            ctgan_score = (
                ctgan._discriminator(
                    torch.as_tensor(ctgan_representation).float().to(device)
                )
                .cpu()
                .detach()
                .numpy()
            )

            acc, auc = compute_metrics_baseline(ctgan_score, Y_test_4baseline)

            X_ref_GLC = ctgan.generate(addition_set.shape[0])

            baseline_results, baseline_scores = baselines(
                X_test_4baseline,
                Y_test_4baseline,
                samples.values,
                addition_set,
                X_ref_GLC,
            )
            baseline_results["hayes"] = {"accuracy": acc, "aucroc": auc}
            baseline_scores["hayes"] = ctgan_score
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Baselines"
            ] = baseline_results
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_BaselineScore"
            ] = baseline_scores
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Xtest"
            ] = X_test_4baseline
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
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
                (p_G_train > thres).sum(0) / training_size,
            )
            print("Eqn.(1), AUC", auc)
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Eqn1"
            ] = (p_G_train > thres).sum(0) / training_size
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Eqn1AUC"
            ] = auc
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
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
                log_p_rel = np.concatenate([p_G_train - p_R_train, p_G_test - p_R_test])
            elif args.density_estimator == "kde":
                log_p_rel = np.concatenate([p_G_train / p_R_train, p_G_test / p_R_test])

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
                    (p_G_train - p_R_train >= thres).sum(0) / training_size,
                )
                print("Eqn.(2), AUC", auc)
                performance_logger[
                    f"{training_size}_{training_epochs}_{held_out_size}"
                ][f"{N_DATA_GEN}_Eqn2"] = (p_G_train - p_R_train > thres).sum(
                    0
                ) / training_size
            elif args.density_estimator == "kde":
                print(
                    "Eqn.(2), training set prediction acc",
                    (p_G_train / p_R_train >= thres).sum(0) / training_size,
                )
                print("Eqn.(2), AUC", auc)
                performance_logger[
                    f"{training_size}_{training_epochs}_{held_out_size}"
                ][f"{N_DATA_GEN}_Eqn2"] = (p_G_train / p_R_train > thres).sum(
                    0
                ) / training_size
            # print('Eqn.(2), test set prediction acc', (p_G_test-p_R_test > thres).sum(0) / training_size)

            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Eqn2AUC"
            ] = auc
            performance_logger[f"{training_size}_{training_epochs}_{held_out_size}"][
                f"{N_DATA_GEN}_Eqn2Score"
            ] = log_p_rel

            print(performance_logger)
