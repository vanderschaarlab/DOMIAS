# future
from __future__ import absolute_import, division, print_function

# stdlib
from pathlib import Path
from typing import Any, Dict, Optional

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import metrics

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.metrics.wd import compute_wd
from domias.models.ctgan import CTGAN
from domias.models.generator import GeneratorInterface

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        continuous: list,
    ) -> None:
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

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z[:, self.feat], self.mean, np.diag(self.var))


def evaluate_performance(
    generator: GeneratorInterface,
    dataset: np.ndarray,
    SIZE_PARAM: int,
    ADDITION_SIZE: int,
    TRAINING_EPOCH: int,
    shifted_column: Optional[int] = None,
    zero_quantile: float = 0.3,
    density_estimator: str = "prior",
    gen_size_list: list = [10000],
    reference_kept_p: float = 1.0,
    seed: int = 0,
    workspace: Path = Path("workspace"),
    device: Any = DEVICE,
) -> Dict:
    performance_logger: Dict = {}

    workspace.mkdir(parents=True, exist_ok=True)

    continuous = []
    for i in np.arange(dataset.shape[1]):
        if len(np.unique(dataset[:, i])) < 10:
            continuous.append(0)
        else:
            continuous.append(1)

    norm = normal_func_feat(dataset, continuous)

    if shifted_column is not None:
        thres = np.quantile(dataset[:, shifted_column], zero_quantile) + 0.01
        dataset[:, shifted_column][dataset[:, shifted_column] < thres] = -999.0
        dataset[:, shifted_column][dataset[:, shifted_column] > thres] = 999.0
        dataset[:, shifted_column][dataset[:, shifted_column] == -999.0] = 0.0
        dataset[:, shifted_column][dataset[:, shifted_column] == 999.0] = 1.0

        training_set = dataset[:SIZE_PARAM]
        print(
            "training data (D_mem) without A=0",
            training_set[training_set[:, shifted_column] == 1].shape,
        )
        training_set = training_set[training_set[:, shifted_column] == 1]

        test_set = dataset[SIZE_PARAM : 2 * SIZE_PARAM]
        test_set = test_set[: len(training_set)]
        addition_set = dataset[-ADDITION_SIZE:]
        addition_set2 = dataset[-2 * ADDITION_SIZE : -ADDITION_SIZE]

        addition_set_A1 = addition_set[addition_set[:, shifted_column] == 1]
        addition_set_A0 = addition_set[addition_set[:, shifted_column] == 0]
        addition_set2_A1 = addition_set2[addition_set2[:, shifted_column] == 1]
        addition_set2_A0 = addition_set2[addition_set2[:, shifted_column] == 0]
        addition_set_A0_kept = addition_set_A0[
            : int(len(addition_set_A0) * reference_kept_p)
        ]
        addition_set2_A0_kept = addition_set2_A0[
            : int(len(addition_set2_A0) * reference_kept_p)
        ]
        if reference_kept_p > 0:
            addition_set = np.concatenate((addition_set_A1, addition_set_A0_kept), 0)
            addition_set2 = np.concatenate((addition_set2_A1, addition_set2_A0_kept), 0)
        else:
            addition_set = addition_set_A1
            addition_set2 = addition_set2_A1
            # test_set = test_set_A1

        SIZE_PARAM = len(training_set)
        ADDITION_SIZE = len(addition_set)

        # hide column A
        training_set = np.delete(training_set, shifted_column, 1)
        test_set = np.delete(test_set, shifted_column, 1)
        addition_set = np.delete(addition_set, shifted_column, 1)
        addition_set2 = np.delete(addition_set2, shifted_column, 1)
        dataset = np.delete(dataset, shifted_column, 1)
    else:
        training_set = dataset[:SIZE_PARAM]
        test_set = dataset[SIZE_PARAM : 2 * SIZE_PARAM]
        addition_set = dataset[-ADDITION_SIZE:]
        addition_set2 = dataset[-2 * ADDITION_SIZE : -ADDITION_SIZE]
    performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"] = {}
    """ 3. Synthesis with TVAE"""
    df = pd.DataFrame(training_set)
    df.columns = [str(_) for _ in range(dataset.shape[1])]

    # Train generator
    print("Train generator")
    generator.fit(df)

    for N_DATA_GEN in gen_size_list:
        print("Sampling from the generator", N_DATA_GEN)
        samples = generator.generate(N_DATA_GEN)
        samples_val = generator.generate(N_DATA_GEN)

        wd_n = min(len(samples), len(addition_set))
        eval_met_on_held_out = compute_wd(samples[:wd_n], addition_set[:wd_n])
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

        np.save(workspace / f"{seed}_synth_samples", samples)
        np.save(workspace / f"{seed}_training_set", training_set)
        np.save(workspace / f"{seed}_test_set", test_set)
        np.save(workspace / f"{seed}_ref_set1", addition_set)
        np.save(workspace / f"{seed}_ref_set2", addition_set2)

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        if density_estimator == "bnaf":
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
                compute_log_p_x(model_gen, torch.as_tensor(test_set).float().to(device))
                .cpu()
                .detach()
                .numpy()
            )
        elif density_estimator == "kde":
            density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
            density_data = stats.gaussian_kde(addition_set.transpose(1, 0))
            p_G_train = density_gen(training_set.transpose(1, 0))
            p_G_test = density_gen(test_set.transpose(1, 0))
        elif density_estimator == "prior":
            density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
            density_data = stats.gaussian_kde(addition_set.transpose(1, 0))
            p_G_train = density_gen(training_set.transpose(1, 0))
            p_G_test = density_gen(test_set.transpose(1, 0))

        X_test_4baseline = np.concatenate([training_set, test_set])
        Y_test_4baseline = np.concatenate(
            [np.ones(training_set.shape[0]), np.zeros(test_set.shape[0])]
        ).astype(bool)
        # build another GAN for hayes and GAN_leak_cal
        ctgan = CTGAN(epochs=TRAINING_EPOCH, pac=1)
        samples.columns = [str(_) for _ in range(dataset.shape[1])]
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
        baseline_results = baseline_results.append(
            {"name": "hayes", "acc": acc, "auc": auc}, ignore_index=True
        )
        baseline_scores["hayes"] = ctgan_score
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

        # eqn1: \prop P_G(x_i)
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
        if density_estimator == "bnaf":
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
            log_p_rel = np.concatenate([p_G_train - p_R_train, p_G_test - p_R_test])
        elif density_estimator == "kde":
            p_R_train = density_data(training_set.transpose(1, 0)) + 1e-30
            p_R_test = density_data(test_set.transpose(1, 0)) + 1e-30
            log_p_rel = np.concatenate([p_G_train / p_R_train, p_G_test / p_R_test])
        elif density_estimator == "prior":
            p_R_train = norm.pdf(training_set) + 1e-30
            p_R_test = norm.pdf(test_set) + 1e-30
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
        if density_estimator == "bnaf":
            print(
                "Eqn.(2), training set prediction acc",
                (p_G_train - p_R_train >= thres).sum(0) / SIZE_PARAM,
            )
            print("Eqn.(2), AUC", auc)
            performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                f"{N_DATA_GEN}_Eqn2"
            ] = (p_G_train - p_R_train > thres).sum(0) / SIZE_PARAM
        elif density_estimator == "kde":
            print(
                "Eqn.(2), training set prediction acc",
                (p_G_train / p_R_train >= thres).sum(0) / SIZE_PARAM,
            )
            print("Eqn.(2), AUC", auc)
            performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                f"{N_DATA_GEN}_Eqn2"
            ] = (p_G_train / p_R_train > thres).sum(0) / SIZE_PARAM
        elif density_estimator == "prior":
            print(
                "Eqn.(2), training set prediction acc",
                (p_G_train / p_R_train >= thres).sum(0) / SIZE_PARAM,
            )
            print("Eqn.(2), AUC", auc)
            performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
                f"{N_DATA_GEN}_Eqn2"
            ] = (p_G_train / p_R_train > thres).sum(0) / SIZE_PARAM

        performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
            f"{N_DATA_GEN}_Eqn2AUC"
        ] = auc
        performance_logger[f"{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}"][
            f"{N_DATA_GEN}_Eqn2Score"
        ] = log_p_rel

    return performance_logger
