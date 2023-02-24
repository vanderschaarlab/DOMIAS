# future
from __future__ import absolute_import, division, print_function

# stdlib
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
    training_size: int,
    held_out_size: int,
    training_epochs: int = 2000,
    synthetic_sizes: list = [10000],
    density_estimator: str = "prior",
    seed: int = 0,
    device: Any = DEVICE,
    shifted_column: Optional[int] = None,
    zero_quantile: float = 0.3,
    reference_kept_p: float = 1.0,
) -> Dict:
    """
    Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
    The provided generator must not be fitted.

    Args:
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

    Returns:
        A dictionary with a key for each of the `synthetic_sizes` values.
        For each `synthetic_sizes` value, the dictionary contains the keys:
            * `MIA_performance` : accuracy and AUCROC for each attack
            * `MIA_scores`: output scores for each attack
            * `data`: the evaluation data
        For both `MIA_performance` and `MIA_scores`, the following attacks are evaluated:
            * "ablated_eq1"
            * "ablated_eq2"
            * "LOGAN_D1"
            * "MC"
            * "gan_leaks"
            * "gan_leaks_cal"
            * "LOGAN_0"
            * "eq1"
            * "domias"
    """
    performance_logger: Dict = {}

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

        training_set = dataset[:training_size]  # membership set
        training_set = training_set[training_set[:, shifted_column] == 1]

        test_set = dataset[training_size : 2 * training_size]  # set of non-members
        test_set = test_set[: len(training_set)]
        reference_set = dataset[-held_out_size:]

        addition_set_A1 = reference_set[reference_set[:, shifted_column] == 1]
        addition_set_A0 = reference_set[reference_set[:, shifted_column] == 0]
        addition_set_A0_kept = addition_set_A0[
            : int(len(addition_set_A0) * reference_kept_p)
        ]
        if reference_kept_p > 0:
            reference_set = np.concatenate((addition_set_A1, addition_set_A0_kept), 0)
        else:
            reference_set = addition_set_A1
            # test_set = test_set_A1

        training_size = len(training_set)
        held_out_size = len(reference_set)

        # hide column A
        training_set = np.delete(training_set, shifted_column, 1)
        test_set = np.delete(test_set, shifted_column, 1)
        reference_set = np.delete(reference_set, shifted_column, 1)
        dataset = np.delete(dataset, shifted_column, 1)
    else:
        training_set = dataset[:training_size]
        test_set = dataset[training_size : 2 * training_size]
        reference_set = dataset[-held_out_size:]

    """ 3. Synthesis with the GeneratorInferface"""
    df = pd.DataFrame(training_set)
    df.columns = [str(_) for _ in range(dataset.shape[1])]

    # Train generator
    generator.fit(df)

    for synthetic_size in synthetic_sizes:
        performance_logger[synthetic_size] = {
            "MIA_performance": {},
            "MIA_scores": {},
            "data": {},
        }
        samples = generator.generate(synthetic_size)
        samples_val = generator.generate(synthetic_size)

        wd_n = min(len(samples), len(reference_set))
        eval_met_on_held_out = compute_wd(samples[:wd_n], reference_set[:wd_n])
        performance_logger[synthetic_size]["MIA_performance"][
            "sample_quality"
        ] = eval_met_on_held_out

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        # First, estimate density of synthetic data
        # BNAF for pG
        if density_estimator == "bnaf":
            _gen, model_gen = density_estimator_trainer(
                samples.values,
                samples_val.values[: int(0.5 * synthetic_size)],
                samples_val.values[int(0.5 * synthetic_size) :],
            )
            _data, model_data = density_estimator_trainer(reference_set)
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
        # KDE for pG
        elif density_estimator == "kde":
            density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_train = density_gen(training_set.transpose(1, 0))
            p_G_test = density_gen(test_set.transpose(1, 0))
        elif density_estimator == "prior":
            density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_train = density_gen(training_set.transpose(1, 0))
            p_G_test = density_gen(test_set.transpose(1, 0))

        X_test_4baseline = np.concatenate([training_set, test_set])
        Y_test_4baseline = np.concatenate(
            [np.ones(training_set.shape[0]), np.zeros(test_set.shape[0])]
        ).astype(bool)

        performance_logger[synthetic_size]["data"]["Xtest"] = X_test_4baseline
        performance_logger[synthetic_size]["data"]["Ytest"] = Y_test_4baseline

        # build another GAN for hayes black-box
        ctgan = CTGAN(epochs=training_epochs, pac=1)
        samples.columns = [str(_) for _ in range(dataset.shape[1])]
        ctgan.fit(samples)  # train a CTGAN on the generated examples

        if ctgan._transformer is None or ctgan._discriminator is None:
            raise RuntimeError()

        # Baselines

        baseline_results, baseline_scores = baselines(
            X_test_4baseline,
            Y_test_4baseline,
            samples.values,
            reference_set,
            reference_set,  # we pass the reference dataset to GAN-leaks CAL for better stability and fairer comparison.
        )

        performance_logger[synthetic_size]["MIA_performance"] = baseline_results
        performance_logger[synthetic_size]["MIA_scores"] = baseline_scores

        # add LOGAN 0 baseline
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

        performance_logger[synthetic_size]["MIA_performance"]["LOGAN_0"] = {
            "accuracy": acc,
            "aucroc": auc,
        }
        performance_logger[synthetic_size]["MIA_scores"]["LOGAN_0"] = ctgan_score

        # Ablated version, based on eqn1: \prop P_G(x_i)
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

        performance_logger[synthetic_size]["MIA_performance"]["eq1"] = {
            "accuracy": (p_G_train > thres).sum(0) / training_size,
            "aucroc": auc,
        }
        performance_logger[synthetic_size]["MIA_scores"]["eq1"] = log_p_test

        # eqn2: \prop P_G(x_i)/P_X(x_i)
        # DOMIAS (BNAF for p_R estimation)
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
        # DOMIAS (KDE for p_R estimation)
        elif density_estimator == "kde":
            p_R_train = density_data(training_set.transpose(1, 0)) + 1e-30
            p_R_test = density_data(test_set.transpose(1, 0)) + 1e-30
            log_p_rel = np.concatenate([p_G_train / p_R_train, p_G_test / p_R_test])
        # DOMIAS (with prior for p_R, see Appendix experiment)
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
            performance_logger[synthetic_size]["MIA_performance"]["domias"] = {
                "accuracy": (p_G_train - p_R_train > thres).sum(0) / training_size,
            }
        elif density_estimator == "kde":
            performance_logger[synthetic_size]["MIA_performance"]["domias"] = {
                "accuracy": (p_G_train / p_R_train > thres).sum(0) / training_size
            }
        elif density_estimator == "prior":
            performance_logger[synthetic_size]["MIA_performance"]["domias"] = {
                "accuracy": (p_G_train / p_R_train > thres).sum(0) / training_size
            }

        performance_logger[synthetic_size]["MIA_performance"]["domias"]["aucroc"] = auc
        performance_logger[synthetic_size]["MIA_scores"]["domias"] = log_p_rel

    return performance_logger
