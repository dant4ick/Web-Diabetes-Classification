"""
This module contains functions for implementing logistic regression, including sigmoid activation function,
parameter initialization, cost computation, gradient descent optimization, and prediction.
"""
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid activation applied to the input array.
    """
    return 1 / (1 + np.exp(-z))


def initialize_parameters(features: int) -> tuple:
    """
    Initialize the weights and bias parameters.

    Args:
        features (int): Number of features in the dataset.

    Returns:
        tuple: Tuple containing the initialized weights and bias (weights, bias).
    """
    return np.zeros((features, 1)), 0


def compute_cost(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """
    Compute the cost function for logistic regression.

    Args:
        X (np.ndarray): Features data.
        y (np.ndarray): Target variable.
        weights (np.ndarray): Weights vector.
        bias (float): Bias term.

    Returns:
        float: Cost value.
    """
    m = len(y)
    predictions = sigmoid(np.dot(X, weights) + bias)
    cost = -1 / m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost


def gradient_descent(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, weights: np.ndarray,
                     bias: float, learning_rate: float, iterations: int) -> tuple:
    """
    Perform gradient descent optimization to find the optimal weights and bias.

    Args:
        X (np.ndarray): Training features data.
        y (np.ndarray): Training target variable.
        X_test (np.ndarray): Test features data.
        y_test (np.ndarray): Test target variable.
        weights (np.ndarray): Initial weights vector.
        bias (float): Initial bias term.
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations for gradient descent.

    Returns:
        tuple: Tuple containing the optimized weights, bias, train costs, and test costs
               (weights, bias, train_costs, test_costs).
    """
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


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Make predictions using the trained logistic regression model.

    Args:
        X (np.ndarray): Features data.
        weights (np.ndarray): Trained weights vector.
        bias (float): Trained bias term.

    Returns:
        np.ndarray: Predictions for the given features data.
    """
    predictions = sigmoid(np.dot(X, weights) + bias)
    return (predictions > 0.5).astype(int)
