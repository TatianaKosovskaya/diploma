#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
url='https://drive.google.com/file/d/1SnMOAOYZUJFR8hUhK58pOU3vLTRaRqvz/view?usp=share_link'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
df.head()
st.table(df.head())
