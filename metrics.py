import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    if (true_positive + false_positive) == 0:
        return 0  # to handle the case where denominator is zero

    precision = true_positive / (true_positive + false_positive)
    return precision


def recall(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if (true_positive + false_negative) == 0:
        return 0  # to handle the case where denominator is zero

    recall = true_positive / (true_positive + false_negative)
    return recall


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if (prec + rec) == 0:
        return 0  # to handle the case where denominator is zero

    f1 = 2 * (prec * rec) / (prec + rec)
    return f1
