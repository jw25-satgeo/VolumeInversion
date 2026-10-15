"""Small, dependency-light regression baselines."""
import numpy as np


def ridge_fit(X, Y, lambdas=None):
    """Fit multi-output ridge regression and select lambda by GCV."""
    lambdas = np.logspace(-6, 3, 40) if lambdas is None else lambdas
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    xm, ym, xs = X.mean(0), Y.mean(0), X.std(0)
    xs[xs < 1e-12] = 1.0
    Xc, Yc = (X - xm) / xs, Y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    n = Xc.shape[0]
    best = (np.inf, None, None)
    for lam in lambdas:
        W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)
        dof = float((s ** 2 / (s ** 2 + lam)).sum())
        gcv = ((Yc - Xc @ W) ** 2).sum() / max(n - dof, 1e-6) ** 2
        if gcv < best[0]:
            best = (gcv, lam, W)
    _, lam, W = best

    def predict(Xn):
        return ((np.asarray(Xn, np.float64) - xm) / xs) @ W + ym

    predict.lam = lam
    return predict
