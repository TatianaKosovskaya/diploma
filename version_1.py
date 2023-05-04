
import streamlit as st
import pandas as pd

st.text('Hi!')

data_1 = pd.read_csv('train_df_1.csv')
data_2 = pd.read_csv('train_df_2.csv')

st.text('Hi 2!')

st.table(data_1.head())
st.table(data_2.head())

st.text('Hi 3!')

data = pd.concat([data_1, data_2], axis=0, ignore_index=True)

st.text('Hi 4!')

st.table(data.tail())

st.text('Hi 5!')
