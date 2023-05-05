
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
import surprise
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic

st.text('Hi!')

data_1 = pd.read_csv('train_df_1.csv')
#data_2 = pd.read_csv('train_df_2.csv')
#data_3 = pd.read_csv('train_df_3.csv')
#data_4 = pd.read_csv('train_df_4.csv')
#data_5 = pd.read_csv('train_df_5.csv')
#data_6 = pd.read_csv('train_df_6.csv')
#data_7 = pd.read_csv('train_df_7.csv')
#data_8 = pd.read_csv('train_df_8.csv')
#data_9 = pd.read_csv('train_df_9.csv')
#data_10 = pd.read_csv('train_df_10.csv')

st.text('Hi 2!')

data = data_1

#data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10], axis=0, ignore_index=True)

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

le = LabelEncoder()
raw_target = data_2.iloc[:, 22:].idxmax(1)
transformed_target = le.fit_transform(raw_target)
data_2['service_opted'] = transformed_target
data_2['service_opted'] = data_2['service_opted'].astype('uint8')
names = raw_target.value_counts().index
values = raw_target.value_counts().values
names = [data_desc[data_desc['Column Name'] == name]['Description'].values[0] for name in names]
user_item_matrix = pd.crosstab(index=data_2.Code, columns=le.transform(raw_target), values=1, aggfunc='sum')
user_item_matrix.fillna(0, inplace=True)
uim_arr = np.array(user_item_matrix)
for row,item in tqdm(enumerate(uim_arr)):
    for column,item_value in enumerate(item):
        uim_arr[row, column] = uim_arr[row, column] / sum(item)
user_item_ratio_matrix = pd.DataFrame(uim_arr, columns=user_item_matrix.columns, index=user_item_matrix.index)
user_item_ratio_stacked = user_item_ratio_matrix.stack().to_frame()
user_item_ratio_stacked['ncodpers'] = [index[0] for index in user_item_ratio_stacked.index]
user_item_ratio_stacked['service_opted'] = [index[1] for index in user_item_ratio_stacked.index]
user_item_ratio_stacked.reset_index(drop=True, inplace=True)
user_item_ratio_stacked.rename(columns={0:"service_selection_ratio"}, inplace=True)
user_item_ratio_stacked = user_item_ratio_stacked[['ncodpers','service_opted', 'service_selection_ratio']]
user_item_ratio_stacked.drop(user_item_ratio_stacked[user_item_ratio_stacked['service_selection_ratio']==0].index, inplace=True)
user_item_ratio_stacked.reset_index(drop=True, inplace=True)
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,1), skip_lines=1)
data = Dataset.load_from_df(user_item_ratio_stacked, reader=reader)
trainset = data.build_full_trainset()
svd = SVD()
svd_results = cross_validate(algo=svd, data=data, cv=4)
svd = SVD()
svd.fit(trainset)
def get_recommendation(uid,model):    
    recommendations = [(uid, sid, data_desc[data_desc['Column Name'] == le.inverse_transform([sid])[0]]['Description'].values[0], model.predict(uid,sid).est) for sid in range(24)]
    recommendations = pd.DataFrame(recommendations, columns=['uid', 'sid', 'service_name', 'pred'])
    recommendations.sort_values("pred", ascending=False, inplace=True)
    recommendations.reset_index(drop=True, inplace=True)
    return recommendations

user_to_remove = []
for index, row in tqdm(enumerate(user_item_matrix.values)):
    non_zeroes = np.count_nonzero(row)
    if non_zeroes < 3:
        user_to_remove.append(user_item_matrix.index[index])
        
user_to_remove = user_item_ratio_stacked[user_item_ratio_stacked['ncodpers'].isin(user_to_remove)].index
user_item_ratio_stacked_reduced = user_item_ratio_stacked.drop(user_to_remove, axis=0, inplace=False)
print(user_item_ratio_stacked_reduced.shape)

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,1), skip_lines=1)
data_reduced = Dataset.load_from_df(user_item_ratio_stacked_reduced, reader=reader)
trainset_reduced = data_reduced.build_full_trainset()

#sim_options = {'name': 'cosine', 'user_based': True}
#sim_user = KNNBasic(sim_options=sim_options, verbose=True, random_state=11)
#sim_user_results = cross_validate(algo=sim_user, data=data_reduced)

sim_options = {'name': 'cosine', 'user_based': True}
sim_user = KNNBasic(sim_options=sim_options, verbose=False, random_state=33)
sim_user.fit(trainset_reduced)



#st.text(svd_results)
st.table(get_recommendation(uid=15890,model=sim_user))

st.text('Hi 6!')

