"""Task 5 guided exercise correction."""
import numpy as np
from scipy.stats import rankdata


def confusion_matrix(y_true, y_pred):
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)

    tn = np.count_nonzero((t == 0) & (p == 0))
    fp = np.count_nonzero((t == 0) & (p == 1))
    fn = np.count_nonzero((t == 1) & (p == 0))
    tp = np.count_nonzero((t == 1) & (p == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def roc_auc_rank(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    ranks = rankdata(scores, method="average")

    n1 = np.count_nonzero(y == 1)
    n0 = np.count_nonzero(y == 0)

    rank_sum = ranks[y == 1].sum()
    u = rank_sum - n1 * (n1 + 1) / 2

    return float(u / (n1 * n0))
