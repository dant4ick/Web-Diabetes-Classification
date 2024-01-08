import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_parameters(features):
    return np.zeros((features, 1)), 0


def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights) + bias)
    cost = -1 / m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost


def gradient_descent(X, y, X_test, y_test, weights, bias, learning_rate, iterations):
    m = len(y)
    train_costs = []
    test_costs = []

    for _ in range(iterations):
        predictions_train = sigmoid(np.dot(X, weights) + bias)
        cost_train = compute_cost(X, y, weights, bias)
        train_costs.append(cost_train)

        dw = 1 / m * np.dot(X.T, (predictions_train - y))
        db = 1 / m * np.sum(predictions_train - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        cost_test = compute_cost(X_test, y_test, weights, bias)
        test_costs.append(cost_test)

    return weights, bias, train_costs, test_costs


def predict(X, weights, bias):
    predictions = sigmoid(np.dot(X, weights) + bias)
    return (predictions > 0.5).astype(int)
