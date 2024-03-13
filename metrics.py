"""
This module contains functions for evaluating the performance of a binary classification model,
including accuracy, precision, recall, F1-score, and plotting the learning curves.
"""
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the precision score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Precision score.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    if (true_positive + false_positive) == 0:
        return 0  # Handle the case where the denominator is zero

    precision = true_positive / (true_positive + false_positive)
    return precision


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the recall score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Recall score.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    if (true_positive + false_negative) == 0:
        return 0  # Handle the case where the denominator is zero

    recall = true_positive / (true_positive + false_negative)
    return recall


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1-score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: F1-score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if (prec + rec) == 0:
        return 0  # Handle the case where the denominator is zero

    f1 = 2 * (prec * rec) / (prec + rec)
    return f1
