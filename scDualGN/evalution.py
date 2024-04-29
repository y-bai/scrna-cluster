import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import homogeneity_score, completeness_score, accuracy_score
from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment

import logging
logger = logging.getLogger(__name__)


def evals(y_true, y_pred, verbosity=True):
    acc = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    hs = homogeneity_score(y_true, y_pred)
    cs = completeness_score(y_true, y_pred)
    purity = cluster_purity(y_true, y_pred)
    if verbosity:
        logger.info('acc {:.4f}, nmi {:.4f}, ari {:.4f}, hs {:.4f}, cs {:.4f}, purity: {:.4f}'.format(acc, nmi, ari, hs, cs, purity))
    return acc, nmi, ari, hs, cs, purity


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1 
    # sklearn v0.22.2
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
    # scipy.optimize for sklearn v1.1.1
    # row_ind, col_ind = linear_assignment(w.max() - w)
    # ind = zip(row_ind, col_ind)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return accuracy_score(y_pred_voted, y_true)
