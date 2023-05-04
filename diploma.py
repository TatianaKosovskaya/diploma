
import streamlit as st
import pandas as pd

data_1 = pd.read_csv('train_df_1.csv')
data_2 = pd.read_csv('train_df_2.csv')
#data_3 = pd.read_csv('train_df_3.csv')
#data_4 = pd.read_csv('train_df_4.csv')
#data_5 = pd.read_csv('train_df_5.csv')
#data_6 = pd.read_csv('train_df_6.csv')
#data_7 = pd.read_csv('train_df_7.csv')
#data_8 = pd.read_csv('train_df_8.csv')
#data_9 = pd.read_csv('train_df_9.csv')
#data_10 = pd.read_csv('train_df_10.csv')

'''
data = data_1.append(data_2, ignore_index=True)
data = data.append(data_3, ignore_index=True)
data = data.append(data_4, ignore_index=True)
data = data.append(data_5, ignore_index=True)
data = data.append(data_6, ignore_index=True)
data = data.append(data_7, ignore_index=True)
data = data.append(data_8, ignore_index=True)
data = data.append(data_9, ignore_index=True)
data = data.append(data_10, ignore_index=True)
'''

data = pd.concat([data_1, data_2], axis=0, ignore_index=True)

st.table(data)

'''
import streamlit as st
import pandas as pd
#import os

st.text('Hi!')

df_1 = pd.read_csv('train_df_1.csv')
st.table(df_1.head())

df_2 = pd.read_csv('train_df_2.csv')
#st.table(df_2.head())

df_3 = pd.read_csv('train_df_3.csv')
df_4 = pd.read_csv('train_df_4.csv')
df_5 = pd.read_csv('train_df_5.csv')
df_6 = pd.read_csv('train_df_6.csv')
df_7 = pd.read_csv('train_df_7.csv')
df_8 = pd.read_csv('train_df_8.csv')
df_9 = pd.read_csv('train_df_9.csv')
df_10 = pd.read_csv('train_df_10.csv')

df_merged = df_1.append(df_2, ignore_index=True)
#st.table(df_merged.tail())
df_merged = df_merged.append(df_3, ignore_index=True)
df_merged = df_merged.append(df_4, ignore_index=True)
df_merged = df_merged.append(df_5, ignore_index=True)
df_merged = df_merged.append(df_6, ignore_index=True)
df_merged = df_merged.append(df_7, ignore_index=True)
df_merged = df_merged.append(df_8, ignore_index=True)
df_merged = df_merged.append(df_9, ignore_index=True)
df_merged = df_merged.append(df_10, ignore_index=True)


st.table(df_merged.head())
st.table(df_merged.tail())
st.table(df_merged)
st.text('Hi!')
'''
