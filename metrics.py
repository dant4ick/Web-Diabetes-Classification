import numpy as np
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


def plot_learning_curves(train_costs, test_costs):
    iterations = len(train_costs)

    plt.plot(range(1, iterations + 1), train_costs, label='Train Cost')
    plt.plot(range(1, iterations + 1), test_costs, label='Test Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()