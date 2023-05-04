
import streamlit as st
import pandas as pd

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
