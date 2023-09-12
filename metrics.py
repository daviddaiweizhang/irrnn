import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score


def mean_squared_difference(true, pred):
    return np.square(true - pred).mean()


def true_positive(true, pred):
    true = ~np.isclose(0, true)
    pred = ~np.isclose(0, pred)
    return pred[true].mean()


def true_negative(true, pred):
    true = np.isclose(0, true)
    pred = np.isclose(0, pred)
    return pred[true].mean()


def true_discovery(true, pred):
    true = ~np.isclose(0, true)
    pred = ~np.isclose(0, pred)
    return true[pred].mean()


def true_omission(true, pred):
    true = np.isclose(0, true)
    pred = np.isclose(0, pred)
    return true[pred].mean()


def false_positive(true, pred):
    return 1 - true_negative(true, pred)


def false_negative(true, pred):
    return 1 - true_positive(true, pred)


def false_discovery(true, pred):
    return 1 - true_discovery(true, pred)


def false_omission(true, pred):
    return 1 - true_omission(true, pred)


def rocauc(true, pred):
    true = ~np.isclose(0, true)
    pred = np.abs(pred)
    return roc_auc_score(true.flatten(), pred.flatten())


def correlation_pearson(true, pred):
    return pearsonr(true, pred)[0]


def correlation_spearman(true, pred):
    return spearmanr(true, pred)[0]


def correlation_kendall(true, pred):
    return kendalltau(true, pred)[0]
