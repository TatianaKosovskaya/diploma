#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd

data_1 = pd.read_csv('train_df_1.csv')
data_2 = pd.read_csv('train_df_2.csv')
data_3 = pd.read_csv('train_df_3.csv')
data_4 = pd.read_csv('train_df_4.csv')
data_5 = pd.read_csv('train_df_5.csv')
data_6 = pd.read_csv('train_df_6.csv')
data_7 = pd.read_csv('train_df_7.csv')
data_8 = pd.read_csv('train_df_8.csv')
data_9 = pd.read_csv('train_df_9.csv')
data_10 = pd.read_csv('train_df_10.csv')
data = df_1.append(df_2, ignore_index=True)
data = data.append(df_3, ignore_index=True)
data = data.append(df_4, ignore_index=True)
data = data.append(df_5, ignore_index=True)
data = data.append(df_6, ignore_index=True)
data = data.append(df_7, ignore_index=True)
data = data.append(df_8, ignore_index=True)
data = data.append(df_9, ignore_index=True)
data = data.append(df_10, ignore_index=True)
st.table(data)

# In[2]:

'''
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")
import surprise
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic
from sklearn.preprocessing import StandardScaler
from numpy import dot
from numpy.linalg import norm
import plotly
import plotly.graph_objs as go


# In[3]:


# Настраиваем отображение максимального количества столбцов
pd.set_option("display.max_columns",100)


# In[4]:


data = pd.read_csv('train_df_s.csv')
data.head()


# In[5]:


data.drop('Unnamed: 0', axis= 1 , inplace= True )
data.head()


# In[6]:


data.rename( columns={'Derivada account':'Derivative account'}, inplace=True )


# In[7]:


l = list(data.columns)
l


# In[8]:


data_desc = pd.read_csv('data_desc.csv')
data_desc


# In[9]:


data_desc['Column Name'] = l
data_desc


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


data.isnull().mean() * 100


# In[14]:


data.isnull().sum()/len(data)*100


# In[15]:


duplicateRows = data[data.duplicated ()]
duplicateRows


# In[16]:


data.describe()


# In[17]:


c_skip = ['Employee index', 'Country residence', 'Sex', 'First date', 'Last date as primary customer', 'Customer type',
         'Customer relation type', 'Residence index', 'Foreigner index', 'Spouse index', 'Channel index', 'Deceased index',
         'Province name', 'Segmentation index']
for c in data.columns:
    print(c, ':')
    print(data[c].unique())
    if c not in c_skip:
        print('Max:', max(data[c].unique()))
        print('Min:', min(data[c].unique()))
    print('', end = '\n\n')


# In[18]:


data_2 = data.copy()
data_2


# In[19]:


data_2 = data_2[data_2['Employee index'].notna()]
data_2 = data_2[data_2['Country residence'].notna()]
data_2 = data_2[data_2['Sex'].notna()]
data_2 = data_2[data_2['Index'].notna()]
data_2 = data_2[data_2['Index real'].notna()]
data_2 = data_2[data_2['Residence index'].notna()]
data_2 = data_2[data_2['Foreigner index'].notna()]
data_2 = data_2[data_2['Deceased index'].notna()]
data_2 = data_2[data_2['Province code'].notna()]
data_2 = data_2[data_2['Province name'].notna()]
data_2 = data_2[data_2['Activity index'].notna()]
data_2 = data_2[data_2['Payroll'].notna()]
data_2 = data_2[data_2['Real pensions'].notna()]


# In[20]:


data_2.shape


# In[21]:


data_2.info()


# In[22]:


data_2.isnull().sum()/len(data_2)*100


# In[24]:


# удалили меньше 1 процента строк
1355499/1364730, len(data_2)/1364730, len(data_2)/len(data)


# In[25]:


data_2.drop(columns=['Last date as primary customer','Spouse index'], inplace=True)
data_2


# In[26]:


data_2.isnull().sum()


# In[27]:


data_2['Gross income of the household']


# In[28]:


round(data_2['Gross income of the household'].mean(), 2)


# In[29]:


data_2['Gross income of the household'] = data_2['Gross income of the household'].fillna(134245.63)


# In[30]:


data_2['Customer type'].mode()


# In[31]:


data_2['Customer type'] = data_2['Customer type'].fillna(1)


# In[32]:


data_2['Customer relation type'].mode()


# In[33]:


data_2['Customer relation type'] = data_2['Customer relation type'].fillna('I')


# In[34]:


data_2['Channel index'].mode()


# In[35]:


data_2['Channel index'] = data_2['Channel index'].fillna('KHE')


# In[36]:


data_2['Segmentation index'].mode()


# In[37]:


data_2['Segmentation index'] = data_2['Segmentation index'].fillna('02 - PARTICULARES')


# In[38]:


data_2.isnull().sum()


# In[ ]:





# In[39]:


data_2.info()


# In[40]:


data_2[['Age', 'Seniority (in months)']] = data_2[['Age', 'Seniority (in months)']].astype(int)
#data_2[['Customer type']] = data_2[['Customer type']].astype(object)


# In[41]:


data_2.info()


# In[42]:


c_skip = ['Customer type']
for c in data_2.columns:
    print(c, ':')
    print(data_2[c].unique())
    if c not in c_skip:
        print('Max:', max(data_2[c].unique()))
        print('Min:', min(data_2[c].unique()))
    print('', end = '\n\n')


# In[ ]:





# In[43]:


sorted(list(data_2['Age'].value_counts().index))


# In[37]:


sorted(list(data_2['Gross income of the household'].value_counts().index), reverse = True)


# In[45]:


data_2 = data_2[data_2["Age"] <= 100]
data_2 = data_2[data_2["Age"] >= 18]
data_2 = data_2[data_2["Seniority (in months)"] != -999999]


# In[46]:


c_skip = ['Customer type']
for c in data_2.columns:
    print(c, ':')
    print(data_2[c].unique())
    if c not in c_skip:
        print('Max:', max(data_2[c].unique()))
        print('Min:', min(data_2[c].unique()))
    print('', end = '\n\n')


# In[40]:


256/12


# In[ ]:





# In[41]:


import plotly
import plotly.graph_objs as go


# In[47]:


data_2.columns


# In[44]:


s_1 = ['Employee index', 'Country residence', 'Sex', 'Index', 'Index real', 'Customer type', 'Customer relation type',
       'Residence index', 'Foreigner index', 'Deceased index', 'Activity index', 'Segmentation index', 'Saving account',
       'Guarantees', 'Current accounts', 'Derivative account', 'Payroll account', 'Junior account', 'Special account', 
       'Particular account', 'Particular plus account', 'Short-term deposits', 'Medium-term deposits', 'Long-term deposits', 
       'E-account', 'Funds', 'Mortgage', 'Planed pensions', 'Loans', 'Taxes', 'Credit card', 'Securities', 'Home account', 
       'Payroll', 'Real pensions', 'Direct debit']
s_2 = ['Age', 'Seniority (in months)', 'Channel index', 'Province code', 'Province name', 'Gross income of the household']
for c in data_2.columns:
    if c in s_1:
        fig = go.Figure()
        fig.add_trace(go.Pie(values=list(data_2[c].value_counts()), 
                             labels=list(data_2[c].value_counts().index), hole=0.9))
        fig.update_layout(
            title=c,
            margin=dict(l=0, r=0, t=30, b=0),
            legend_orientation="h")
        fig.show()
    if c in s_2:
        fig = go.Figure(data=[go.Histogram(x=data_2[c])])
        fig.update_layout(
            title=c,
            title_x = 0.5,
            xaxis_title=c,
            yaxis_title="Number of people",
            legend=dict(x=.5, xanchor="center", orientation="h"),
            margin=dict(l=0, r=0, t=30, b=0))
        fig.show()


# In[68]:


import numpy as np


# In[69]:


# Исправление категорий столбца - indrel_1mes
data_2['Customer type'].replace('1', 1, inplace=True)
data_2['Customer type'].replace('1.0', 1, inplace=True)
data_2['Customer type'].replace('2', 2, inplace=True)
data_2['Customer type'].replace('2.0', 2, inplace=True)
data_2['Customer type'].replace('3', 3, inplace=True)
data_2['Customer type'].replace('3.0', 3, inplace=True)
data_2['Customer type'].replace('4', 4, inplace=True)
data_2['Customer type'].replace('4.0', 4, inplace=True)
data_2['Customer type'].replace('P', 5, inplace=True)
data_2['Customer type'].replace('None',np.nan, inplace=True)

# Вывод на экран части данных
data_2.head()


# In[ ]:





# # Импорт библиотек

# In[29]:


# Для манипулирования данными
import numpy as np
import pandas as pd
from datetime import datetime

# Для графиков
import seaborn as sns
import matplotlib.pyplot as plt

# Для предварительной обработки данных
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import cosine_similarity

# Модели машинного обучения, используемые для заполнения нулевых значений
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# Для визуализации итераций
from tqdm import tqdm

# Для чтения файлов и данных
import os

# Игнорировать предупреждения
import warnings
warnings.filterwarnings("ignore")


# In[30]:


# Настраиваем отображение максимального количества столбцов
pd.set_option("display.max_columns",1000)


# # Чтение тренировочного датасета

# In[31]:


# Чтение описания данных, которое будет использовано позже
data_desc = pd.read_csv("data_desc.csv")


# In[32]:


#new
data_desc


# # Кодирование цели для выдачи рекомендаций

# In[70]:


# Определить объект кодировщика метки
le = LabelEncoder()

# Преобразование векторов с one-hot кодированием в один столбец
raw_target = data_2.iloc[:, 22:].idxmax(1)

# Использование Fit для трансформирования меток
transformed_target = le.fit_transform(raw_target)

# Объединить столбец с фреймворком данных
data_2['service_opted'] = transformed_target

# Присвойте тип uint8 для экономии памяти
data_2['service_opted'] = data_2['service_opted'].astype('uint8')

# Вывести на экран фрейм данных
data_2.head(10)


# In[71]:


# Проверка количества продуктов
plt.figure(figsize=(12,8))

# Получить имя и вхождения
names = raw_target.value_counts().index
values = raw_target.value_counts().values

# Сопоставить имена с их английским переводом через data_desc
names = [data_desc[data_desc['Column Name'] == name]['Description'].values[0] for name in names]

# Построить график
ax = sns.barplot(x=names, y=values)

# Установить заголовок
ax.set_title("Number Of Services Opted In Millions")

# Установить xticklabels и повернуть
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# Подписать столбцы
for p in ax.patches:
    ax.annotate("{:.1f}".format(p.get_height()), (p.get_x(), p.get_height()), rotation=25)

# Отображение графика на экране
plt.show()


# # Создание матрицы взаимодействия пользователя с элементом, представляющей количество

# In[72]:


# Создавая матрицу пользовательских элементов, каждая запись указывает, сколько раз услуга выбиралась этим пользователем.
user_item_matrix = pd.crosstab(index=data_2.Code, columns=le.transform(raw_target), values=1, aggfunc='sum')

# Заполнение значений nan как 0, так как служба не выбрана
user_item_matrix.fillna(0, inplace=True)

# Вывести матрицу пользовательских элементов (представляет количество)
user_item_matrix


# # Создание матрицы взаимодействия пользователя с элементом, представляющей соотношение

# In[73]:


# Преобразование user_item_matrix в тип данных массива
uim_arr = np.array(user_item_matrix)

# Перебираем каждую строку (пользователь)
for row,item in tqdm(enumerate(uim_arr)):
    # Перебираем каждый столбец (элемент)
    for column,item_value in enumerate(item):
        # Изменяем количество выбранных услуг на отношение
        uim_arr[row, column] = uim_arr[row, column] / sum(item)
        
# Конвертируем массив в dataframe для лучшего просмотра
user_item_ratio_matrix = pd.DataFrame(uim_arr, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Выводим на экран user_item_ratio_matrix (представляет соотношение)
user_item_ratio_matrix


# # Складываем в один столбец

# In[74]:


# Складываем user_item_ratio_matrix, чтобы получить все значения в одном столбце
user_item_ratio_stacked = user_item_ratio_matrix.stack().to_frame()

# Создаем столбец для идентификатора пользователя
user_item_ratio_stacked['ncodpers'] = [index[0] for index in user_item_ratio_stacked.index]

# Создаем столбец для service_opted
user_item_ratio_stacked['service_opted'] = [index[1] for index in user_item_ratio_stacked.index]

# Сбрасываем и удаляем индекс
user_item_ratio_stacked.reset_index(drop=True, inplace=True)

# Выводим на экран фрейм данных
user_item_ratio_stacked


# # Правильное представление данных

# In[75]:


# Переименовываем столбец 0 в service_selection_ratio
user_item_ratio_stacked.rename(columns={0:"service_selection_ratio"}, inplace=True)

# Располагаем столбец систематически для лучшего обзора
user_item_ratio_stacked = user_item_ratio_stacked[['ncodpers','service_opted', 'service_selection_ratio']]

# Отбросываем все строки с 0 записями, так как это означает, что пользователь никогда не выбирал услугу
user_item_ratio_stacked.drop(user_item_ratio_stacked[user_item_ratio_stacked['service_selection_ratio']==0].index, inplace=True)

# Сбрасываем индекс
user_item_ratio_stacked.reset_index(drop=True, inplace=True)

# Показываем окончательный фрейм данных
user_item_ratio_stacked


# # Совместная фильтрация — на основе модели

# # Surprise

# In[39]:


import surprise

from surprise import Dataset, Reader

from surprise.prediction_algorithms.matrix_factorization import SVD

from surprise.model_selection import cross_validate

from surprise import accuracy


# # Создание совместимого набора данных Surprise

# In[76]:


# Инициализация объект-сюрприза для чтения
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,1), skip_lines=1)

# Загрузка данных
data = Dataset.load_from_df(user_item_ratio_stacked, reader=reader)

# Построение объекта из набора тренировочных данных (это выполняется только тогда, когда необходимо использовать весь набор 
# данных для обучения)
trainset = data.build_full_trainset()


# # Выполнение перекрестной проверки

# In[77]:


# Инициализация модели
svd = SVD()

# перекрестная проверка
svd_results = cross_validate(algo=svd, data=data, cv=4)

# Полученные результаты!
svd_results


# # Создание и обучение модели

# In[42]:


# Инициализация модели
svd = SVD()

# перекрестная проверка
svd.fit(trainset)


# # Делаем прогнозы/рекомендации

# In[78]:


def get_recommendation(uid,model):    
    recommendations = [(uid, sid, data_desc[data_desc['Column Name'] == le.inverse_transform([sid])[0]]['Description'].values[0], model.predict(uid,sid).est) for sid in range(24)]
    # Преобразование в данные pandas
    recommendations = pd.DataFrame(recommendations, columns=['uid', 'sid', 'service_name', 'pred'])
    # Сортировка по pred
    recommendations.sort_values("pred", ascending=False, inplace=True)
    # Сброс индексов
    recommendations.reset_index(drop=True, inplace=True)
    # Возвращаем результат
    return recommendations


# In[79]:


get_recommendation(15890,svd)


# # Совместная фильтрация — на основе памяти

# # Удаление пользователей, купивших как минимум 3 разные услуги

# In[80]:


# Печать формы и кадра данных коэффициента стека df
print(user_item_ratio_stacked.shape)
user_item_ratio_stacked.head()


# In[81]:


# Пустой список пользователей для удаления
user_to_remove = []

for index, row in tqdm(enumerate(user_item_matrix.values)):
    # Подсчитаем количество ненулевых элементов
    non_zeroes = np.count_nonzero(row)
    # Проверяем, что non_zeros меньше 3
    if non_zeroes < 3:
        # Добавляем идентификатор пользователя в список
        user_to_remove.append(user_item_matrix.index[index])


# In[82]:


# Получаем индекс из user_item_ratio_stacked, где существует user_to_del
user_to_remove = user_item_ratio_stacked[user_item_ratio_stacked['ncodpers'].isin(user_to_remove)].index

# Удаляем элементы из user_item_ratio_stacked
user_item_ratio_stacked_reduced = user_item_ratio_stacked.drop(user_to_remove, axis=0, inplace=False)

# Печатаем результат
print(user_item_ratio_stacked_reduced.shape)
user_item_ratio_stacked_reduced.head()


# # Создать набор данных, совместимый с Surprise

# In[83]:


# Инициализировать объект-сюрприз для чтения
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,1), skip_lines=1)

# Загружаем данные
data_reduced = Dataset.load_from_df(user_item_ratio_stacked_reduced, reader=reader)

# Строим объект набора тренировочных данных (это выполняется только тогда, когда необходимо использовать весь набор данных для 
# обучения)
trainset_reduced = data_reduced.build_full_trainset()


# # Импорт библиотек

# In[84]:


from surprise.prediction_algorithms.knns import KNNBasic


# # Выполнение перекрестной проверки

# In[85]:


# Объявление параметров сходства.
sim_options = {'name': 'cosine',
               'user_based': True}

# Алгоритм KNN используется для поиска похожих элементов
sim_user = KNNBasic(sim_options=sim_options, verbose=True, random_state=11)

# перекрестная проверка
sim_user_results = cross_validate(algo=sim_user, data=data_reduced, cv=4)

# Полученные результаты!
sim_user_results


# # Настройка и обучение модели

# In[86]:


# Объявление параметров сходства.
sim_options = {'name': 'cosine',
               'user_based': True}

# Алгоритм KNN используется для поиска похожих элементов
sim_user = KNNBasic(sim_options=sim_options, verbose=False, random_state=33)

# Обучаем алгоритм на наборе тренировочных данных и предсказываем оценки для набора тестов
sim_user.fit(trainset_reduced)


# # Делаем прогнозы/рекомендации

# In[87]:


get_recommendation(uid=1226375.0,model=sim_user)


# In[88]:


get_recommendation(uid=15890.0,model=sim_user)


# # Система рекомендаций на основе памяти предметов

# # Выполнение перекрестной проверки

# In[89]:


# Объявление параметров сходства.
sim_options = {'name': 'cosine',
               'user_based': False}

# Алгоритм KNN используется для поиска похожих элементов
sim_item = KNNBasic(sim_options=sim_options, verbose=False, random_state=33)

# перекрестная проверка
sim_item_results = cross_validate(algo=sim_item, data=data, cv=4)

# Полученные результаты!
sim_item_results


# # Настройка и обучение модели

# In[90]:


# Объявление параметров сходства.
sim_options = {'name': 'cosine',
               'user_based': False}

# Алгоритм KNN используется для поиска похожих элементов
sim_item = KNNBasic(sim_options=sim_options, verbose=False, random_state=33)

# Обучаем алгоритм на наборе тренировочных данных и предсказываем оценки для набора тестов
sim_item.fit(trainset)


# # Делаем прогнозы/рекомендации

# In[91]:


get_recommendation(1553685.0, sim_item)


# In[92]:


get_recommendation(1226375.0, sim_item)


# In[93]:


get_recommendation(15890.0, sim_item)


# # Система рекомендаций на основе демографии и активности

# # Кодирование категориальных переменных

# In[95]:


# Список столбцов для кодирования
cols_to_encode = ['Employee index', 'Country residence', 'Sex', 'Index real', 'Customer relation type', 'Residence index', 
                  'Foreigner index', 'Channel index', 'Deceased index', 'Segmentation index']

# Список энкодеров меток, которые позже будут использоваться для преобразований
label_encoders = []

# Create Label итеративно кодирует эти столбцы
for col in tqdm(cols_to_encode):
    # Инициализация объекта кодировщика меток
    lab_enc = LabelEncoder()
    
    # Кодируем столбец и заменяем его существующим
    data_2[col] = lab_enc.fit_transform(data_2[col])
    
    # Преобразование типов в uint8 dtype
    data_2[col] = data_2[col].astype('uint8')
    
    # Добавление его в список label_encoders, чтобы использовать позже
    label_encoders.append(lab_enc)
    
    # Удаление объекта кодировщика этикетки
    del lab_enc
    
# Вывод данных на экран
data_2.head()


# In[96]:


# Удаляем столбец nomprov, так как у нас уже есть его закодированная функция (cod_prov)
data_2.drop(columns=['Province name'], inplace=True)

# Удаление столбца tipodom, так как все значения равны '1'
data_2.drop(columns=['Addres type'], inplace=True)

# Вывод данных
data_2.head()


# # Подготовка набора данных для рекомендательной системы

# # 1. Выбираем последнюю транзакцию для каждого пользователя

# In[97]:


# Выбор неповторяющихся строк (уникальных) и сохранение последней транзакции с помощью параметра keep='last'
user_data = data_2[~data_2['Code'].duplicated(keep='last')]

# Сброс индекса
user_data.reset_index(drop=True, inplace=True)

# Выводим на экран шапку данных
user_data.head()


# # 2. Создаем столбец, в котором хранится количество услуг, ранее использованных пользователем

# In[98]:


from tqdm.notebook import tqdm
tqdm.pandas()


# In[99]:


# Создание одноразовых кодировок с использованием переменных service_opted
service_one_hot = pd.get_dummies(user_data['service_opted'],prefix='service')

# Присоединение к сервису one hot с реальными данными
user_data = pd.concat([user_data, service_one_hot], axis=1)

# Вывод на экран части данных
user_data.head()


# In[100]:


# Установка идентификатора пользователя и сервиса, выбранного в качестве индекса
data_2.set_index(['Code','service_opted'], inplace=True)

# Сортируем индекс, чтобы быстрее получать записи
data_2.sort_index(inplace=True)

# Выводим фрейм данных
data_2.head()


# # Создание функции с one-hot кодированием, хранящим количество сервисов, имеющихся у пользователя

# In[101]:


# Список сервисных меток
service_list = [i for i in range(24)]

# Для каждой сервисной метки
for service_no in tqdm(service_list):
    # Перебираем каждую строку user_data
    for index, row in tqdm(enumerate(user_data.itertuples())):
        # Получаем счетчик старых транзакций текущего пользователя
        try:
            old_service_no_count = data_2.loc[(row.Code, service_no)].shape[0]
        except:
            old_service_no_count = 0
        finally:
            # Создаем новые столбцы и добавляем в них данные
            user_data.at[index, f'service_{service_no}'] = old_service_no_count
        
# Выводим на экран фрейм данных user_data
user_data.head()


# # Создание функции с использованием столбцов fecha_alta и fecha_dato

# In[103]:


# Создание функции Fecha Alto
user_data['First date day'] = user_data['First date'].progress_apply(lambda date: datetime(list(map(int, date.split('-')))[0], list(map(int, date.split('-')))[1], list(map(int, date.split('-')))[2]).weekday())
user_data['First date month'] = user_data['First date'].progress_apply(lambda date: int(date.split('-')[1]))
user_data['First date year'] = user_data['First date'].progress_apply(lambda date: int(date.split('-')[0]))

# Преобразование всех этих столбцов в uint8 (диапазон 0-255), кроме года, для экономии памяти, так как эти функции будут 
# в этом диапазоне
user_data['First date day'] = user_data['First date day'].astype('uint8')
user_data['First date month'] = user_data['First date month'].astype('uint8')
user_data['First date year'] = user_data['First date year'].astype('int16')

# Удаляем столбец fecha_alta
del user_data['First date'], user_data['Date']

# Выводим на экран часть данных
user_data.head()


# # Разделить набор данных как Target и Feature

# In[104]:


Y = user_data['service_opted'].copy()
X = user_data.drop(columns=['service_opted'], inplace=False)


# # Масштабирование набора данных

# In[105]:


from sklearn.preprocessing import StandardScaler


# In[106]:


# Определяем масштабирующий объект
scaler = StandardScaler()

# Подходящее преобразование данных
user_data_scaled = scaler.fit_transform(X)


# # Выполнение уменьшения размерности

# In[107]:


from sklearn.decomposition import PCA


# In[109]:


# Определяем экземпляр PCA
pca = PCA(0.95)

# Подходящее преобразование данных
user_data_reduced = pd.DataFrame(pca.fit_transform(user_data_scaled), index=user_data.Code)

# Вывод данных
user_data_reduced.head()


# # Получение рекомендации для указанного пользователя

# In[110]:


from numpy import dot
from numpy.linalg import norm


# In[111]:


def get_label_name(label, le=le):
    return data_desc[data_desc['Column Name'] == le.inverse_transform([label])[0]]['Description'].values[0]


# In[112]:


def cosine_sim(X,Y):
    return dot(X,Y) / (norm(X)*norm(Y))


# In[113]:


def get_sim_user_recommendation(uid, top_n, X):
    # Получение указанного пользователя
    user_specified = X.loc[uid]
    
    # Рассчитываем сходство с каждым пользователем
    res = X.progress_apply(lambda user: cosine_sim(user_specified, user), axis=1)
    
    # Преобразовываем в фрейм данных
    res = res.to_frame(name='sim_score')
    
    # Удаляем индекс и делаем его столбцом
    res.reset_index(inplace=True)
    
    # Присоединяем пользовательские данные и красную таблицу к энкодерам
    res = pd.merge(left= user_data[['Code','service_opted']], right = res, on='Code')
    
    # Выбраем наиболее похожую строку из каждой категории услуг
    res = res[~res['service_opted'].duplicated(keep='first')]
    
    # Сортируем результаты
    res.sort_values(by='sim_score', ascending=False, inplace=True)
    
    # Добавляем столбец с выбранным именем службы
    res['service_opted_name'] = res['service_opted'].progress_apply(lambda label: get_label_name(label, le))
    
    # Удаляем индекс и делаем его столбцом
    res.reset_index(drop=True, inplace=True)
    
    # Возвращаем предсказания
    return res


# # Проверка рекомендаций

# In[115]:


# Получить результат для 55890.0 (82 г/0)
res1 = get_sim_user_recommendation(55890.0, 24, user_data_reduced)
res1


# In[116]:


# Получить результат для 891565.0 (возраст-51)
res2 = get_sim_user_recommendation(891565.0, 24, user_data_reduced)
res2


# In[117]:


data_2[data_2['Age'] == 22]


# In[118]:


# Получить результат для 174986.0 (возраст-22)
res3 = get_sim_user_recommendation(174986.0, 24, user_data_reduced)
res3
'''
