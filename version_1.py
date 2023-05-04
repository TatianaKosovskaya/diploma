
import streamlit as st
import pandas as pd

data_1 = pd.read_csv('train_df_1.csv')

st.table(data_1)

