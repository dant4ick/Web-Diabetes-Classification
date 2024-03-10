"""
This is the main entry point for the logistic regression model training and evaluation.
It loads the dataset, performs data preprocessing, trains the logistic regression model,
evaluates its performance, and visualizes the learning curves.
"""
from ucimlrepo import fetch_ucirepo
from data_processing import train_test_split, standardize_data, balance_dataset
from logistic_regression import initialize_parameters, gradient_descent, predict
from metrics import accuracy, precision, recall, f1_score, plot_learning_curves


def main():
    """
    Main function to run the logistic regression model training and evaluation.
    """
    # Fetch the dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # Get the data as pandas dataframes
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # Print the metadata
    print(cdc_diabetes_health_indicators.metadata)

    # Print the variable information
    print(cdc_diabetes_health_indicators.variables)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Balance training data
    X_train, y_train = balance_dataset(X_train, y_train)

    # Standardize the data
    X_train, X_test = standardize_data(X_train, X_test)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Initialize parameters
    features = X_train.shape[1]
    weights, bias = initialize_parameters(features)

    # Hyperparameters
    learning_rate = 0.05
    iterations = 300

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
