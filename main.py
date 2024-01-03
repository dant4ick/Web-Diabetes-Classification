from translation import translate_dataset
from data_processing import check_missing_values, replace_missing_values, one_hot_encoding, train_test_split, \
    standardize_data
from logistic_regression import initialize_parameters, gradient_descent, predict
from metrics import accuracy, precision, recall, f1_score

translate_dataset('data/mushroom_dataset.csv')
features_to_transform = check_missing_values('data/mushroom_dataset_ru.csv')
replace_missing_values('data/mushroom_dataset_ru.csv', features_to_transform)
one_hot_encoding('data/data_with_replacements.csv')

X_train, X_test, y_train, y_test = train_test_split('data/encoded_data.csv', 'съедобность_съедобен')
X_train, X_test = standardize_data(X_train, X_test)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

features = X_train.shape[1]
weights, bias = initialize_parameters(features)

# Гиперпараметры логистической регрессии
learning_rate = 0.1
iterations = 100

# Обучение модели
weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate, iterations)

# Предсказание на тестовых данных
predictions = predict(X_test, weights, bias)

print(f'accuracy: {accuracy(y_test, predictions)}')
print(f'precision: {precision(y_test, predictions)}')
print(f'recall: {recall(y_test, predictions)}')
print(f'f1_score: {f1_score(y_test, predictions)}')
