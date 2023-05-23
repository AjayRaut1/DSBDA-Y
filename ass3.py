#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('iris1.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[12]:


np.mean(df['sepal_length'])


# In[13]:


np.mean(df)


# In[14]:


np.std(df)


# In[15]:


np.min(df)


# In[16]:


np.max(df)


# In[17]:


df.quantile(0.25)

