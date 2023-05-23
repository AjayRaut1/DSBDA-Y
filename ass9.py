#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


titanic = sns.load_dataset('titanic')


# In[3]:


titanic


# # we will use train.csv 

# In[4]:


# df=pd.read_csv('train.csv')
df=titanic
df


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


sns.countplot(x=df['survived'])


# In[13]:


sns.countplot(x=df['sex'])


# In[14]:


df['sex'].value_counts().plot(kind = 'pie', autopct='%.2f')


# In[15]:


df['survived'].value_counts().plot(kind = 'pie', autopct='%.2f')


# In[16]:


sns.barplot(x=df['survived'],y=df['age'])


# In[17]:


sns.barplot(x=df['sex'],y=df['survived'])


# In[18]:


sns.barplot(x=df['sex'],y=df['age'],hue=df['survived'])


# In[19]:


sns.boxplot(x=df['sex'],y=df['age'])


# In[20]:


sns.boxplot(x=df['sex'],y=df['age'],hue=df['survived'])


# In[21]:


pd.crosstab(df['sex'],df['survived'])


# In[22]:


pd.crosstab(df['age'],df['survived'])


# In[23]:


sns.heatmap(pd.crosstab(df['sex'],df['survived']))


# In[24]:


sns.heatmap(pd.crosstab(df['age'],df['survived']))


# In[25]:


sns.clustermap(pd.crosstab(df['sex'],df['survived']))


# In[26]:


sns.clustermap(pd.crosstab(df['age'],df['survived']))

