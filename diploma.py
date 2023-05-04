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
data = data_1.append(data_2, ignore_index=True)
data = data.append(data_3, ignore_index=True)
data = data.append(data_4, ignore_index=True)
data = data.append(data_5, ignore_index=True)
data = data.append(data_6, ignore_index=True)
data = data.append(data_7, ignore_index=True)
data = data.append(data_8, ignore_index=True)
data = data.append(data_9, ignore_index=True)
data = data.append(data_10, ignore_index=True)
st.table(data)

