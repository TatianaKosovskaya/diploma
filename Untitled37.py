#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
#import os

st.text('Hi!')

df_1 = pd.read_csv('train_df_1.csv')
st.table(df_1.head())

df_2 = pd.read_csv('train_df_2.csv')
st.table(df_2.head())

df_merged = df_1.append(df_2, ignore_index=True)
st.table(df_merged.head())
st.table(df_merged.tail())

import glob
import os
# merging the files
joined_files = os.path.join("train_df_*.csv")
  
# A list of all joined files is returned
joined_list = glob.glob(joined_files)
  
# Finally, the files are joined
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
st.table(df.head())
