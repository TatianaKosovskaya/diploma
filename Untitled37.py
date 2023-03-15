#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[7]:


url='https://drive.google.com/file/d/1lnBoSa6wV5cb8k996hXhePGqtkEsxHbN/view?usp=share_link'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
df.head()

