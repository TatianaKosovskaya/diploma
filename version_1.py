
import streamlit as st

st.text('Hi_!')

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
#import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from numpy import dot
from numpy.linalg import norm

st.text('Hi!')

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

st.text('Hi 2!')

data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10], axis=0, ignore_index=True)

st.text('Hi 3!')

st.table(data.tail())

st.text('Hi 4!')

data.drop('Unnamed: 0', axis= 1 , inplace= True )
data.rename( columns={'Derivada account':'Derivative account'}, inplace=True )
l = list(data.columns)
data_desc = pd.read_csv('data_desc.csv')
data_desc['Column Name'] = l
data_2 = data.copy()
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
data_2.drop(columns=['Last date as primary customer','Spouse index'], inplace=True)
data_2['Gross income of the household'] = data_2['Gross income of the household'].fillna(134245.63)
data_2['Customer type'] = data_2['Customer type'].fillna(1)
data_2['Customer relation type'] = data_2['Customer relation type'].fillna('I')
data_2['Channel index'] = data_2['Channel index'].fillna('KHE')
data_2['Segmentation index'] = data_2['Segmentation index'].fillna('02 - PARTICULARES')
data_2[['Age', 'Seniority (in months)']] = data_2[['Age', 'Seniority (in months)']].astype(int)
data_2 = data_2[data_2["Age"] <= 100]
data_2 = data_2[data_2["Age"] >= 18]
data_2 = data_2[data_2["Seniority (in months)"] != -999999]
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

st.table(data_2.head())

st.text('Hi 5!')

'''
import surprise
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic
'''
