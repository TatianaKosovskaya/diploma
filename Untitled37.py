#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd

df_1 = pd.read_csv('train_df_1.csv')
df_1.head()
st.table(df_1.head())

df_2 = pd.read_csv('train_df_2.csv')
df_2.head()
st.table(df_2.head())
