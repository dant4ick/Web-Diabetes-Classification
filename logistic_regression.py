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


def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights) + bias)
        dw = 1 / m * np.dot(X.T, (predictions - y))
        db = 1 / m * np.sum(predictions - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias


def predict(X, weights, bias):
    predictions = sigmoid(np.dot(X, weights) + bias)
    return (predictions > 0.5).astype(int)
