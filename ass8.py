#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df


# In[4]:


sns.distplot(df['pclass'])


# In[5]:


df


# In[6]:


sns.distplot(df['age'], bins=40)


# In[7]:


sns.distplot(df['age'],bins=20, kde=False, rug=True)


# In[8]:


sns.jointplot(x=df['age'],y=df['fare'],kind='scatter')


# In[9]:


sns.jointplot(x=df['age'],y=df['fare'],kind='hex')


# In[10]:


sns.pairplot(df)


# In[11]:


sns.barplot(x=df['sex'],y=df['fare'])


# In[12]:


sns.countplot(x=df['pclass'])


# In[13]:


sns.distplot(df['fare'],hist=True)

