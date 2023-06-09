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
#st.table(df_2.head())

df_3 = pd.read_csv('train_df_3.csv')
df_4 = pd.read_csv('train_df_4.csv')
df_5 = pd.read_csv('train_df_5.csv')
df_6 = pd.read_csv('train_df_6.csv')
df_7 = pd.read_csv('train_df_7.csv')
df_8 = pd.read_csv('train_df_8.csv')
df_9 = pd.read_csv('train_df_9.csv')
df_10 = pd.read_csv('train_df_10.csv')
#df_11 = pd.read_csv('train_df_11.csv')
#df_12 = pd.read_csv('train_df_12.csv')
#df_13 = pd.read_csv('train_df_13.csv')
#df_14 = pd.read_csv('train_df_14.csv')
#df_15 = pd.read_csv('train_df_15.csv')
#df_16 = pd.read_csv('train_df_16.csv')
#df_17 = pd.read_csv('train_df_17.csv')
#df_18 = pd.read_csv('train_df_18.csv')
#df_19 = pd.read_csv('train_df_19.csv')
#df_20 = pd.read_csv('train_df_20.csv')

data = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10], axis=0, ignore_index=True)
st.table(data)

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
#df_merged = df_merged.append(df_11, ignore_index=True)
#df_merged = df_merged.append(df_12, ignore_index=True)
#df_merged = df_merged.append(df_13, ignore_index=True)
#df_merged = df_merged.append(df_14, ignore_index=True)
#df_merged = df_merged.append(df_15, ignore_index=True)
#df_merged = df_merged.append(df_16, ignore_index=True)
#df_merged = df_merged.append(df_17, ignore_index=True)
#df_merged = df_merged.append(df_18, ignore_index=True)
#df_merged = df_merged.append(df_19, ignore_index=True)
#df_merged = df_merged.append(df_20, ignore_index=True)

st.table(df_merged.head())
st.table(df_merged.tail())
st.text('Hi!')

#import glob
#import os
# merging the files
#joined_files = os.path.join("train_df_*.csv")
  
# A list of all joined files is returned
#joined_list = glob.glob(joined_files)
  
# Finally, the files are joined
#df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
#st.table(df.head())
