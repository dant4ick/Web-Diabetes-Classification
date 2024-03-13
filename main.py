"""
This is the main entry point for the logistic regression model training and evaluation.
It loads the dataset, performs data preprocessing, trains the logistic regression model,
evaluates its performance, visualizes the learning curves, and plots various visualizations
related to the data and model.
"""
from ucimlrepo import fetch_ucirepo
from data_processing import train_test_split, standardize_data, balance_dataset, apply_pca
from logistic_regression import initialize_parameters, gradient_descent, predict
from metrics import accuracy, precision, recall, f1_score
from plots import plot_learning_curves, plot_correlation_matrix, plot_pca_explained_variance_ratio, plot_pca_scatter, \
    plot_pca_variance


def main():
    """
    Main function to run the logistic regression model training and evaluation.
    """
    # Fetch the dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # Get the data as pandas dataframes
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # Plot the correlation matrix
    plot_correlation_matrix(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Reducing the dimension of the original dataset to 8 components
    X_train, X_test = apply_pca(X_train, X_test, n_components=8)

    # Plot the cumulative explained variance ratio for PCA
    plot_pca_explained_variance_ratio(X_train)

    # Plot the PCA scatter plot
    plot_pca_scatter(X_train)

    # Plot the variances of PCA components
    plot_pca_variance(X_train)

    # Balance data
    X_train, y_train = balance_dataset(X_train, y_train)
    # X_test, y_test = balance_dataset(X_test, y_test)

    # Standardize the data
    X_train, X_test = standardize_data(X_train, X_test)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Initialize parameters
    features = X_train.shape[1]
    weights, bias = initialize_parameters(features)

    # Hyperparameters
    learning_rate = 0.1
    iterations = 1000

    # Train the model
    weights, bias, train_costs, test_costs = gradient_descent(X_train, y_train, X_test, y_test, weights, bias,
                                                              learning_rate, iterations)

    # Predict on the test set
    predictions = predict(X_test, weights, bias)

    # Evaluate the model
    print(f'Accuracy: {accuracy(y_test, predictions)}')
    print(f'Precision: {precision(y_test, predictions)}')
    print(f'Recall: {recall(y_test, predictions)}')
    print(f'F1-score: {f1_score(y_test, predictions)}')

    # Plot the learning curves
    plot_learning_curves(train_costs, test_costs)


if __name__ == '__main__':
    main()
