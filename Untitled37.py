#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd


# In[7]:


#url='https://drive.google.com/file/d/1lnBoSa6wV5cb8k996hXhePGqtkEsxHbN/view?usp=share_link'
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#df = pd.read_csv(path)
#st.table(df.head())

#https://github.com/TatianaKosovskaya/diploma/blob/main/mushrooms.csv
#url='https://github.com/TatianaKosovskaya/diploma/blob/main/mushrooms.csv'
df = pd.read_csv('mushrooms.csv')
st.table(df.head())
