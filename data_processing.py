import pandas as pd
import numpy as np


def check_missing_values(input_file):
    # Чтение CSV файла
    df = pd.read_csv(input_file)

    # Подсчет пропущенных значений для каждого признака
    missing_values = df.isnull().sum()

    # Список признаков с пропущенными значениями и соответствующих most_common_value
    features_with_missing_values = []

    # Вывод информации о пропущенных значениях
    for column, count in missing_values.items():
        if count != 0:
            print(f"\nПризнак: {column}")
            print(f"Количество пропущенных значений: {count}")

            # Доля пропущенных значений в этом признаке (%)
            percentage_missing = (count / len(df)) * 100
            print(f"Доля пропущенных значений: {percentage_missing:.2f}%")

            # Самое популярное значение среди заполненных для этого признака
            most_common_value = df[column].mode()[0]

            # Доля этого значения от всех значений (%)
            percentage_most_common = (df[column].value_counts()[most_common_value] / len(df)) * 100
            print(f"Самое популярное значение среди заполненных: {most_common_value}")
            print(f"Доля самого популярного значения: {percentage_most_common:.2f}%")

            # Добавление в список
            features_with_missing_values.append((column, most_common_value))

    return features_with_missing_values


def replace_missing_values(input_file, replacements):
    # Чтение CSV файла
    df = pd.read_csv(input_file)

    # Замена пропущенных значений
    for feature, replacement_value in replacements:
        df[feature].fillna(replacement_value, inplace=True)

    # Сохранение обновленных данных
    df.to_csv('data/data_with_replacements.csv', index=False, encoding='utf-8')


def one_hot_encoding(input_file):
    # Чтение CSV файла
    df = pd.read_csv(input_file)

    # Список категориальных столбцов
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Кодирование категориальных данных
    for column in categorical_columns:
        # Создание новых столбцов для каждой уникальной категории
        unique_values = df[column].unique()
        for value in unique_values:
            new_column_name = f"{column}_{value}"
            df[new_column_name] = (df[column] == value).astype(int)

        # Удаление исходного категориального столбца и одного бинарного столбца
        df = df.drop([column, f"{column}_{unique_values[0]}"], axis=1)

    # Запись обновленных данных в CSV файл
    df.to_csv('data/encoded_data.csv', index=False, encoding='utf-8')


def train_test_split(input_file, target_column, test_size=0.2, random_state=69):
    df = pd.read_csv(input_file)

    # Разделение данных на факторы (X) и целевую переменную (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    np.random.seed(random_state)

    # Перемешиваем индексы
    shuffled_indices = np.random.permutation(len(X))

    # Вычисляем количество элементов в тестовой выборке
    test_size = int(len(X) * test_size)

    # Индексы для обучающей и тестовой выборок
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Разделяем данные
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    # Вычисляем среднее значение и стандартное отклонение для каждого признака в обучающей выборке
    mean_values = X_train.mean()
    std_dev_values = X_train.std()

    # Стандартизация обучающей выборки
    X_train_standardized = (X_train - mean_values) / std_dev_values

    # Стандартизация тестовой выборки (используя среднее и стандартное отклонение обучающей выборки)
    X_test_standardized = (X_test - mean_values) / std_dev_values

    return X_train_standardized, X_test_standardized