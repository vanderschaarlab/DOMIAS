# third party
import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.preprocessing import MinMaxScaler


def compute_wd(
    X_syn: np.ndarray,
    X: np.ndarray,
) -> float:
    X_ = X.copy()
    X_syn_ = X_syn.copy()
    if len(X_) > len(X_syn_):
        X_syn_ = np.concatenate(
            [X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))]
        )

    scaler = MinMaxScaler().fit(X_)

    X_ = scaler.transform(X_)
    X_syn_ = scaler.transform(X_syn_)

    X_ten = torch.from_numpy(X_).reshape(-1, 1)
    Xsyn_ten = torch.from_numpy(X_syn_).reshape(-1, 1)
    OT_solver = SamplesLoss(loss="sinkhorn")

    return OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()
