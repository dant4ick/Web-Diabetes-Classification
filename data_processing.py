"""
This module contains functions for data preprocessing tasks such as splitting the data into train and test sets,
standardizing the data, and oversampling the minority class using the SMOTE technique.
"""
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 69) -> tuple:
    """
    Split the data into train and test sets.

    Args:
        X (pd.DataFrame): Features data.
        y (pd.Series): Target variable.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Seed for the random number generator. Defaults to 69.

    Returns:
        tuple: Tuple containing the train and test sets (X_train, X_test, y_train, y_test).
    """
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(len(X))

    # Calculate the number of elements in the test set
    test_size = int(len(X) * test_size)

    # Indices for train and test sets
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the data
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize the data by subtracting the mean and dividing by the standard deviation.

    Args:
        X_train (pd.DataFrame): Training features data.
        X_test (pd.DataFrame): Test features data.

    Returns:
        tuple: Tuple containing the standardized training and test sets (X_train_standardized, X_test_standardized).
    """
    # Calculate the mean and standard deviation for each feature in the training set
    mean_values = X_train.mean()
    std_dev_values = X_train.std()

    # Standardize the training set
    X_train_standardized = (X_train - mean_values) / std_dev_values

    # Standardize the test set (using the mean and standard deviation from the training set)
    X_test_standardized = (X_test - mean_values) / std_dev_values

    return X_train_standardized, X_test_standardized


def balance_dataset(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Balance the dataset using SMOTE oversampling technique.

    Args:
        X (pd.DataFrame): Features data.
        y (pd.Series): Target variable.

    Returns:
        tuple: Oversampled features and target data (X_resampled, y_resampled).
    """
    # Create a SMOTE object
    smote = SMOTE()

    # Apply SMOTE oversampling
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Class distribution before oversampling:", Counter(y))
    print("Class distribution after oversampling:", Counter(y_resampled))

    return X_resampled, y_resampled
