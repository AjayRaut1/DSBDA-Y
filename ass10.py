#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv('iris.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


sns.distplot(df['sepal.length'])


# In[10]:


sns.distplot(df['sepal.width'])


# In[11]:


sns.distplot(df['petal.length'])


# In[12]:


sns.distplot(df['petal.width'])


# In[13]:


plt.hist(df['sepal.length'])


# In[14]:


plt.hist(df['sepal.width'])


# In[15]:


plt.hist(df['petal.length'])


# In[16]:


plt.hist(df['petal.width'])


# In[17]:


sns.boxplot(df['sepal.width'])


# In[18]:


sns.boxplot(df['sepal.length'])


# In[19]:


sns.boxplot(df['petal.length'])


# In[20]:


sns.boxplot(df['petal.width'])

