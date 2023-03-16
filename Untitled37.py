#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import zipfile
zf = zipfile.ZipFile('train_ver2.csv.zip')
df = pd.read_csv(zf.open('train_ver2.csv'))
st.table(df.head())
