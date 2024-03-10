from ucimlrepo import fetch_ucirepo
from pprint import pprint


def check_class_balance(data, target_column):
    """
    Функция для проверки сбалансированности классов и получения информации о распределении классов.

    Args:
        data (pandas.DataFrame): Набор данных в формате DataFrame.
        target_column (str): Имя столбца, содержащего целевую переменную.

    Returns:
        dict: Словарь с информацией о балансе классов.
    """
    class_counts = data[target_column].value_counts()
    total_instances = len(data)

    class_balance_info = {}

    for class_label, count in class_counts.items():
        class_balance_info[class_label] = {
            'count': count,
            'percentage': (count / total_instances) * 100,
            'imbalance_ratio': total_instances / count
        }

    # Вычисляем коэффициент дисбаланса классов
    majority_class_count = class_counts.max()
    minority_class_count = class_counts.min()
    class_imbalance_ratio = majority_class_count / minority_class_count

    class_balance_info['class_imbalance_ratio'] = class_imbalance_ratio

    return class_balance_info


# Пример использования
data = fetch_ucirepo(id=891).data
class_balance_info = check_class_balance(data, 'targets')
pprint(class_balance_info)
