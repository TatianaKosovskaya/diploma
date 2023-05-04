
st.text('Hi_!')

import streamlit as st
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
import surprise
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic
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

