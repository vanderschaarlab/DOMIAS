# stdlib
from typing import Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        return np.sum((X - Y) ** 2, axis=1)
    else:
        res = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in X:
            res[i] = d(x, Y)

        return res


def d_min(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.min(d(X, Y))


def GAN_leaks(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    scores = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        scores[i] = np.exp(-d_min(x, X_G))
    return scores


def GAN_leaks_cal(X_test: np.ndarray, X_G: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    # Actually, they retrain a generative model to sample X_ref from.
    # This doesn't seem necessary to me and creates an additional, unnecessary
    # dependence on (and noise from) whatever model is used
    scores = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        scores[i] = np.exp(-d_min(x, X_G) + d_min(x, X_ref))
    return scores


def hayes(X_test: np.ndarray, X_G: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    num = min(X_G.shape[0], X_ref.shape[0])
    # can use auxiliary data model, i.e. already implemented
    # they show it doesn't work well
    # full black box uses GAN.
    # naive classifier trained on generative and real data
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=1000
    ).fit(
        np.vstack([X_G[:num], X_ref[:num]]),
        np.concatenate([np.ones(num), np.zeros(num)]),
    )
    #                                                           np.zeros(X_ref.shape[0])]))
    return clf.predict_proba(X_test)[:, 1]


def hayes_torch(X_test: np.ndarray, X_G: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    num = min(X_G.shape[0], X_ref.shape[0])

    class Net(torch.nn.Module):
        def __init__(
            self, input_dim: int, hidden_dim: int = 256, out_dim: int = 2
        ) -> None:
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            out = self.fc3(x)
            return out

    batch_size = 256
    clf = Net(input_dim=X_test.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    all_x, all_y = np.vstack([X_G[:num], X_ref[:num]]), np.concatenate(
        [np.ones(num), np.zeros(num)]
    )
    all_x = torch.as_tensor(all_x).float().to(DEVICE)
    all_y = torch.as_tensor(all_y).long().to(DEVICE)
    X_test = torch.as_tensor(X_test).float().to(DEVICE)
    for training_iter in range(int(300 * len(X_test) / batch_size)):
        rnd_idx = np.random.choice(len(X_test), batch_size)
        train_x, train_y = all_x[rnd_idx], all_y[rnd_idx]
        clf_out = clf(train_x)
        loss = loss_func(clf_out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return_out = clf(X_test)[:, 1].cpu().detach().numpy()
    torch.cuda.empty_cache()
    return return_out


def hilprecht(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    scores = np.zeros(X_test.shape[0])
    distances = np.zeros((X_test.shape[0], X_G.shape[0]))
    for i, x in enumerate(X_test):
        distances[i] = d(x, X_G)
    # median heuristic (Eq. 4 of Hilprecht)
    min_dist = np.min(distances, 1)
    assert min_dist.size == X_test.shape[0]
    epsilon = np.median(min_dist)
    for i, x in enumerate(X_test):
        scores[i] = np.sum(distances[i] < epsilon)
    scores = scores / X_G.shape[0]
    return scores


def kde_baseline(
    X_test: np.ndarray, X_G: np.ndarray, X_ref: np.ndarray
) -> Tuple[float, float]:
    # Eq. 1
    p_G_approx = stats.gaussian_kde(X_G.transpose(1, 0))
    score_1 = p_G_approx.evaluate(X_test.transpose(1, 0))

    # Eq. 2
    p_R_approx = stats.gaussian_kde(X_ref.transpose(1, 0))
    score_2 = score_1 / (p_R_approx.evaluate(X_test.transpose(1, 0)) + 1e-20)

    # score_3 = score_1/(p_R_approx.evaluate(X_test.transpose(1,0)))
    return score_1, score_2  # , score_3


def compute_metrics_baseline(
    y_scores: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    # if len(np.unique(y_scores))<=2: # we don't want binarized scores
    #    raise ValueError('y_scores should contain non-binarized values, but only contains', np.unique(y_scores))
    y_pred = y_scores > np.median(y_scores)
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    return acc, auc


def baselines(
    X_test: np.ndarray,
    Y_test: np.ndarray,
    X_G: np.ndarray,
    X_ref: np.ndarray,
    X_ref_GLC: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, dict]:
    score = {}
    score["Eq. 1"], score["Eq. 2"] = kde_baseline(X_test, X_G, X_ref)
    score["hayes_torch"] = hayes_torch(X_test, X_G, X_ref)
    score["hilprecht"] = hilprecht(X_test, X_G)
    score["GAN-leaks"] = GAN_leaks(X_test, X_G)
    score["GAN-leaks_cal"] = GAN_leaks_cal(X_test, X_G, X_ref_GLC)
    results = pd.DataFrame(columns=["name", "acc", "auc"])
    for name, y_scores in score.items():
        try:
            acc, auc = compute_metrics_baseline(
                y_scores, Y_test, sample_weight=sample_weight
            )
            results = pd.concat(
                [results, pd.DataFrame({"name": name, "acc": acc, "auc": auc})],
                ignore_index=True,
            )
        except BaseException:
            np.save("temp_debug_scores", y_scores)
    return results, score
