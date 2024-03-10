# main.py
from ucimlrepo import fetch_ucirepo
from data_processing import check_missing_values, replace_missing_values, one_hot_encoding, train_test_split, standardize_data
from logistic_regression import initialize_parameters, gradient_descent, predict
from metrics import accuracy, precision, recall, f1_score, plot_learning_curves

# Fetch the dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# Get the data as pandas dataframes
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Print the metadata
print(cdc_diabetes_health_indicators.metadata)

# Print the variable information
print(cdc_diabetes_health_indicators.variables)

# Check for missing values
features_to_transform = check_missing_values(X)

# Replace missing values
X = replace_missing_values(X, features_to_transform)

# One-hot encoding for categorical features
X = one_hot_encoding(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

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
weights, bias, train_costs, test_costs = gradient_descent(X_train, y_train, X_test, y_test, weights, bias, learning_rate, iterations)

# Predict on the test set
predictions = predict(X_test, weights, bias)

# Evaluate the model
print(f'Accuracy: {accuracy(y_test, predictions)}')
print(f'Precision: {precision(y_test, predictions)}')
print(f'Recall: {recall(y_test, predictions)}')
print(f'F1-score: {f1_score(y_test, predictions)}')

# Plot the learning curves
plot_learning_curves(train_costs, test_costs)
