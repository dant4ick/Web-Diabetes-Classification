# Применение градиентного спуска для обучения логистической регрессионной модели машинного обучения в задаче прогнозирования диабета

В этом проекте реализована модель логистической регрессии для прогнозирования диабета с
использованием набора данных из открытого репозитория машинного обучения UCI. Проект включает в себя компоненты
предварительной
обработки данных, обучения модели, оценки и визуализации.

### Модули

- `data_processing.py`: Содержит функции для задач предварительной обработки данных, таких как разделение данных на
  обучающие и тестовые наборы, стандартизация данных и передискретизация класса меньшинства с использованием метода
  SMOTE.
- `logistic_regression.py`: Реализует модель логистической регрессии, включая функцию активации,
  инициализацию параметров, обучение методом градиентного спуска и прогнозирование.
- `main.py`: Основная точка входа в проект. Он загружает набор данных, выполняет предварительную обработку данных,
  обучает модель
  логистической регрессии, оценивает ее производительность и генерирует различные визуализации.
- `metrics.py`: Определяет функции для оценки производительности модели бинарной классификации, включая accuracy,
  precision, recall, и F1-меру.
- `plots.py`: Содержит функции для создания визуализаций, таких как кривые обучения, корреляционные матрицы и
  Графики, связанные с PCA.

### Установка

Для установки зависимостей для данного проекта вам понадобится установленный Python 3 и следующие пакеты:

```
pip install numpy pandas scikit-learn matplotlib imbalanced-learn
```

### Использование

Чтобы запустить проект, просто выполните скрипт `main.py`:

```
python main.py
```

### Описание

Данный проект является форком [ClassificationProject](https://github.com/LyoshaGodX/ClassificationProject) применительно
к другому набору данных. Модель логистической регрессии, градиентный спуск а также метрики производительности модели
были реализованы с нуля. PCA, SMOTE а также корреляционная матрица были добавлены с использованием
библиотек `numpy`, `sklearn` и `imblearn`.

### Результаты

![Corr.png](images%2FCorr.png)

Матрица корреляций показывает корреляцию между целевой переменной (Diabetes_binary) и различными признаками в исходном
наборе данных. Более темные ячейки указывают на более сильную положительную корреляцию, а более светлые - на более
сильную отрицательную корреляцию. Мы видим, что признаки, такие как `HighBP`, `HighChol` и `CholCheck`, имеют
относительно высокую положительную корреляцию с наличием диабета, в то время как `Income` и `Education` имеют
отрицательную корреляцию.

Исходный набор данных содержит 21 признак и 1 целевую переменную. В целях повышение производительности модели по времени
исполнения был реализован алогоритм уменьшения размерности данных PCA.

![Component.png](images%2FComponent.png)

Этот график показывает дисперсии (variance) первых 8 главных компонент PCA (Principal Component Analysis) после
преобразования исходных данных. Мы видим, что первая главная компонента имеет наибольшую дисперсию, что означает, что
она объясняет наибольшую долю вариации в данных. Дисперсии последующих компонент постепенно уменьшаются. Это типичный
паттерн для PCA, где первые несколько компонент объясняют большую часть изменчивости, а остальные компоненты вносят
меньший вклад.

![Variance.png](images%2FVariance.png)

Этот график иллюстрирует долю общей дисперсии в данных, объясняемую каждой из первых 8 главных компонент ПКА. Мы видим,
что первая компонента объясняет около 60% дисперсии, а первые 4 компоненты вместе объясняют примерно 80% дисперсии. Это
означает, что, используя эти 4 компоненты, мы можем сохранить большую часть информации, содержащейся в исходных данных,
при этом значительно уменьшив размерность данных.

Было принято решение сократить размерность набора данных до 8 компонент.

![Scatter.png](images%2FScatter.png)

Этот график представляет собой диаграмму рассеяния, где каждая точка соответствует одному наблюдению, спроецированному
на первые две главные компоненты PCA. Мы видим, что точки образуют несколько кластеров, что указывает на присутствие
различных групп или паттернов в данных. В данном случае мы видим несколько отчетливых перьев или кластеров, указывающих
на сильную корреляцию между некоторыми признаками и целевой переменной. Также можно сделать вывод о том, что
экстремальных значений немного и процесс предобработки данных можно продолжать.

Набор данных является несбалансированным, около 80% всех наблюдений представляют собой случай с целевой переменной в
значении 1, то есть наличие диабета. Для устранения дисбаланса класса используем метод генерации синтетических данных
для соблюдения пропорции классов 50/50. Для этого используется метод SMOTE, функция `balance_dataset` из
файла [data_processing.py](data_processing.py).

Применяя метод градиентного спуска обученная модель ведет себя следующим образом:

![Learning.png](images%2FLearning.png)

График показывает кривые обучения модели логистической регрессии. Мы видим, что стоимость (cost) на обучающем наборе
данных быстро уменьшается с увеличением числа итераций градиентного спуска. Стоимость на тестовом наборе также
уменьшается, но не так быстро, как на обучающем.

### Метрики производительности модели

| Метрика   | Значение |
|-----------|----------|
| Accuracy  | 0.7154   |
| Precision | 0.2928   |
| Recall    | 0.7358   |
| F1-score  | 0.4189   |

Эти метрики показывают, что модель логистической регрессии достигла достаточно высоких результатов в задаче бинарной
классификации на этом наборе данных. Accuracy показывает, что модель верно предсказывает 71% всех наблюдений в наборе
данных. При этом показатель Precision достаточно низок, он отвечает за количество ложноположительных предсказаний, то
есть модель склонна предсказывать диабет там, где его нет на самом деле. Однако Recall значительно выше, чем в модели
без применения предобработки данных, он отвечает за количество ложноотрицательных предсказаний. Таким образом, модель
делает крайне мало предсказаний, в которых человек с диабетом получит прогноз о том, что у него диабета нет. Это очень
важная и положительная характеристика модели. F1-score показывает гармоническое среднее между Precision и Recall, его
высокое значние сигнализирует о качестве модели с учетом высокого Recall.







