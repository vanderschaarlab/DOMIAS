from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

plt.close("all")
np.random.seed(22)


def data_loader(name="housing"):
    if name == "healthcare":
        X = pd.read_csv("healthcare.csv").to_numpy()[:, 1:-1]
    # np.random.multivariate_normal([0],[[1]], n_M)*std1 # non-training data
    elif name == "housing":
        X = fetch_california_housing().data

    np.random.shuffle(X)
    scaler = StandardScaler()
    for i in range(X.shape[1]):
        if len(np.unique(X[:, i])) > 2:
            scaler = StandardScaler()
            X[:, i] = scaler.fit_transform(X[:, i].reshape(-1, 1)).squeeze()
        else:
            pass  # X[:,i] = X[:,i].astype(float)
    return X.astype(float)


def d(X, Y):
    if len(X.shape) == 1:
        return np.sum((X - Y) ** 2, axis=1)
    else:
        res = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in X:
            res[i] = d(x, Y)

        return res


def d_min(X, Y):
    return np.min(d(X, Y))


def GAN_leaks(X_test, X_G):
    scores = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        scores[i] = np.exp(-d_min(x, X_G))
    return scores


def GAN_leaks_cal(X_test, X_G, X_ref):
    # Actually, they retrain a generative model to sample X_ref from.
    # This doesn't seem necessary to me and creates an additional, unnecessary
    # dependence on (and noise from) whatever model is used
    scores = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        scores[i] = np.exp(-d_min(x, X_G) + d_min(x, X_ref))
    return scores


def hayes(X_test, X_G, X_ref):
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


def hayes_torch(X_test, X_G, X_ref):
    num = min(X_G.shape[0], X_ref.shape[0])
    import torch
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=256, out_dim=2):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            out = self.fc3(x)
            return out

    import numpy as np

    batch_size = 256
    clf = Net(input_dim=X_test.shape[1]).cuda()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    all_x, all_y = np.vstack([X_G[:num], X_ref[:num]]), np.concatenate(
        [np.ones(num), np.zeros(num)]
    )
    all_x = torch.as_tensor(all_x).float().cuda()
    all_y = torch.as_tensor(all_y).long().cuda()
    X_test = torch.as_tensor(X_test).float().cuda()
    for training_iter in range(int(300 * len(X_test) / batch_size)):
        rnd_idx = np.random.choice(len(X_test), batch_size)
        train_x, train_y = all_x[rnd_idx], all_y[rnd_idx]
        clf_out = clf(train_x)
        loss = loss_func(clf_out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.cpu().detach().item())
    return_out = clf(X_test)[:, 1].cpu().detach().numpy()
    torch.cuda.empty_cache()
    return return_out


def hilprecht(X_test, X_G):
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


def kde_baseline(X_test, X_G, X_ref):
    # Eq. 1
    p_G_approx = stats.gaussian_kde(X_G.transpose(1, 0))
    score_1 = p_G_approx.evaluate(X_test.transpose(1, 0))

    # Eq. 2
    p_R_approx = stats.gaussian_kde(X_ref.transpose(1, 0))
    score_2 = score_1 / (p_R_approx.evaluate(X_test.transpose(1, 0)) + 1e-20)

    # score_3 = score_1/(p_R_approx.evaluate(X_test.transpose(1,0)))
    return score_1, score_2  # , score_3


def compute_metrics_baseline(y_scores, y_true, sample_weight=None):
    # if len(np.unique(y_scores))<=2: # we don't want binarized scores
    #    raise ValueError('y_scores should contain non-binarized values, but only contains', np.unique(y_scores))
    y_pred = y_scores > np.median(y_scores)
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    return acc, auc


def baselines(X_test, Y_test, X_G, X_ref, X_ref_GLC, sample_weight=None):
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
            results = results.append(
                {"name": name, "acc": acc, "auc": auc}, ignore_index=True
            )
        except BaseException:
            print("name", name)
            np.save("temp_debug_scores", y_scores)
    #     thres = np.quantile(score['Eq. 3'], 0.5)
    #     acc = ((score['Eq. 3'] >= thres) == Y_test.astype(int)).sum(0) / Y_test.shape[0]
    #     fpr, tpr, thresholds = metrics.roc_curve(Y_test.astype(int), score['Eq. 3'], pos_label=1)
    #     auc = metrics.auc(fpr, tpr)
    #     results = results.append({'name':'Eq. 2 H', 'acc':acc, 'auc': auc}, ignore_index=True)
    return results, score


def main(name="housing"):
    n_M = 500  # number of training samples
    n_G = 5000  # number of generated samples
    n_ref = 5000  # number of reference samples (for Eq. 2)

    # create real data
    X_all = data_loader(name)[: n_M * 2 + n_ref]

    X_M = X_all[:n_M]  # training data
    X_nM = X_all[n_M : 2 * n_M]  # non-trainin test data
    X_ref = X_all[2 * n_M :]  # reference data

    # draws from kde approximation
    p_G = stats.gaussian_kde(X_M.transpose(1, 0))
    X_G = p_G.resample(n_G).transpose(1, 0)

    # make binary again
    if name == "healthcare":
        for i in range(X_all.shape[1]):
            if len(np.unique(X_all[:, i])) <= 2:
                X_G[:, i] = np.round(X_G[:, i])

    X_test = np.concatenate([X_M, X_nM])
    Y_test = np.concatenate([np.ones(X_M.shape[0]), np.zeros(X_nM.shape[0])]).astype(
        bool
    )

    results = baselines(X_test, Y_test, X_G, X_ref)

    if name == "housing":
        thres = np.quantile(X_all[:, 0], 0.95)
        select = X_test[:, 0] > thres
    else:
        select = X_test[:, 15] * X_test[:, 16]

    results_outliers = baselines(X_test, Y_test, X_G, X_ref, sample_weight=select)

    print(results)
    plt.bar(range(len(res)), results["acc"], tick_label=results["name"])

    print("outlier results:")
    plt.hist(X_test[:, 0], 20)
    print(results_outliers)
    plt.bar(
        range(len(results_outliers)),
        results["acc"],
        tick_label=results_outliers["name"],
    )
    return results


if __name__ == "__main__":
    name = "housing"
    res = main(name)
