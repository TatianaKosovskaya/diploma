#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st


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

# Настраиваем отображение максимального количества столбцов
pd.set_option("display.max_columns",1000)
# In[7]:


#url='https://drive.google.com/file/d/1lnBoSa6wV5cb8k996hXhePGqtkEsxHbN/view?usp=share_link'
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#df = pd.read_csv(path)
#st.table(df.head())

#https://github.com/TatianaKosovskaya/diploma/blob/main/mushrooms.csv
#url='https://github.com/TatianaKosovskaya/diploma/blob/main/mushrooms.csv'
df = pd.read_csv('mushrooms.csv')
st.table(df.head())
