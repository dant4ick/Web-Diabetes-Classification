from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np


def check_missing_values(X):
    # Count missing values for each feature
    missing_values = X.isnull().sum()

    # List of features with missing values and their most common values
    features_with_missing_values = []

    # Print information about missing values
    for column, count in missing_values.items():
        if count != 0:
            print(f"\nFeature: {column}")
            print(f"Number of missing values: {count}")

            # Percentage of missing values in this feature
            percentage_missing = (count / len(X)) * 100
            print(f"Percentage of missing values: {percentage_missing:.2f}%")

            # Most common value among the filled values for this feature
            most_common_value = X[column].mode()[0]

            # Percentage of the most common value among all values
            percentage_most_common = (X[column].value_counts()[most_common_value] / len(X)) * 100
            print(f"Most common value among filled values: {most_common_value}")
            print(f"Percentage of the most common value: {percentage_most_common:.2f}%")

            # Add to the list
            features_with_missing_values.append((column, most_common_value))

    return features_with_missing_values


def replace_missing_values(X, replacements):
    # Replace missing values
    for feature, replacement_value in replacements:
        X[feature].fillna(replacement_value, inplace=True)

    return X


def one_hot_encoding(X):
    # List of categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Encode categorical data
    for column in categorical_columns:
        # Create new columns for each unique category
        unique_values = X[column].unique()
        for value in unique_values:
            new_column_name = f"{column}_{value}"
            X[new_column_name] = (X[column] == value).astype(int)

        # Drop the original categorical column and one binary column
        X = X.drop([column, f"{column}_{unique_values[0]}"], axis=1)

    return X


def train_test_split(X, y, test_size=0.2, random_state=69):
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


def standardize_data(X_train, X_test):
    # Calculate the mean and standard deviation for each feature in the training set
    mean_values = X_train.mean()
    std_dev_values = X_train.std()

    # Standardize the training set
    X_train_standardized = (X_train - mean_values) / std_dev_values

    # Standardize the test set (using the mean and standard deviation from the training set)
    X_test_standardized = (X_test - mean_values) / std_dev_values

    return X_train_standardized, X_test_standardized


def balance_dataset(X, y):
    """
    Balance the dataset using SMOTE oversampling technique.

    Args:
        X (pandas.DataFrame): Features data.
        y (pandas.Series): Target variable.

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
