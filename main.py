import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Функция для построения тепловой карты
def plot_heatmap(data, title):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.show()

def normalize_data():
    # Нормализация данных
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data_filled), columns=data_filled.columns)
    data_normalized.to_csv("analize/norm_file_path.csv", sep=';', index=False)

def graph_rass(columns):
    # Построение диаграмм рассеяния для каждой переменной по отношению к "ИЗВ"
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data_filled[column], y=data_filled[target_variable])
        plt.title(f'Диаграмма рассеяния: {column} vs {target_variable}')
        plt.xlabel(column)
        plt.ylabel(target_variable)
        plt.grid(True)
        plt.show()

def to_csv_intr():
    data_filled.to_csv("analize/interpolation_file_path.csv", sep=';')

def data_graphs(data_filled):
    csv_file_path = 'data/data.csv'
    data_csv = pd.read_csv(csv_file_path)

    data_filled = pd.concat([data_filled, data_csv["Дата отбора"]], axis=1)
    print(data_filled)

    # Преобразование колонки 'Дата отбора' в формат datetime
    data_filled['Дата отбора'] = pd.to_datetime(data_filled['Дата отбора'], errors='coerce')

    # Установка 'Дата отбора' как индекса
    data_filled.set_index('Дата отбора', inplace=True)

    # Выбор показателей для построения графиков
    indicators = ['Фосфаты, мг/л', 'Индекс ITS', 'Растворённый_кислород, %', 'БПК5, мгО2/л']

    # Построение временных графиков для каждого показателя
    plt.figure(figsize=(14, 10))

    for i, indicator in enumerate(indicators):
        plt.subplot(2, 2, i + 1)
        plt.plot(data_filled.index, data_filled[indicator], marker='o', linestyle='-', markersize=3)
        plt.title(indicator)
        plt.xlabel('Дата отбора')
        plt.ylabel(indicator)
        plt.grid(True)

    plt.tight_layout()
    plt.show()


target_variable = 'ИЗВ'
file_path = 'data/Сводная Суздальские (1).csv'  # Укажите путь к вашему файлу
df = pd.read_csv(file_path, delimiter=';')

df = df.apply(pd.to_numeric, errors='coerce')
# Замена NaN на средние значения по каждому столбцу
data_filled = df.interpolate()

features = data_filled.drop(columns=[target_variable])
half_index = len(features.columns) // 2

# Первая половина признаков
first_half_features = features.iloc[:, :half_index]
# Вторая половина признаков
second_half_features = features.iloc[:, half_index:]

plot_heatmap(pd.concat([first_half_features, data_filled[target_variable]], axis=1),
             'Тепловая карта корреляции - Первая половина признаков')

# Построение тепловой карты для второй половины признаков с 'ИЗВ'
plot_heatmap(pd.concat([second_half_features, data_filled[target_variable]], axis=1),
             'Тепловая карта корреляции - Вторая половина признаков')

# features = data_filled.drop(columns=[target_variable])
# half_index = len(features.columns) // 2
#
# # Первая половина признаков
# first_half_features = features.iloc[:, :half_index]
# # Вторая половина признаков
# second_half_features = features.iloc[:, half_index:]
#
# plot_heatmap(pd.concat([first_half_features, data_filled[target_variable]], axis=1),
#              'Тепловая карта корреляции - Первая половина признаков')
#
# # Построение тепловой карты для второй половины признаков с 'ИЗВ'
# plot_heatmap(pd.concat([second_half_features, data_filled[target_variable]], axis=1),
#              'Тепловая карта корреляции - Вторая половина признаков')






