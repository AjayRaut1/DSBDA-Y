#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('StudentsPerformance.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


df['math_score']=df['math_score'].interpolate()


# In[9]:


df.isnull().sum()


# In[10]:


df['writing_score']=df['writing_score'].fillna(method='ffill')


# In[11]:


df.isnull().sum()


# In[12]:


df['reading_score']=df['reading_score'].fillna(method='bfill')


# In[13]:


df.isnull().sum()


# In[14]:


df


# In[15]:


df.dtypes


# In[25]:


df['math_score']=df['math_score'].astype('int64')
df['reading_score']=df['reading_score'].astype('int64')
df['writing_score']=df['writing_score'].astype('int64')
df.dtypes


# In[17]:


columns=['math_score','reading_score','writing_score']
df.boxplot(columns)


# In[26]:


np.where(df['math_score']>60)


# In[27]:


np.where(df['math_score']<80)


# In[20]:


df


# In[21]:


df[(df['math_score']>60)&(df['math_score']<80)]


# In[22]:


df[(df['math_score']>60)&(df['reading_score']<80)]


# In[23]:


new_df1=df[
          ((df.math_score>=60)&(df.math_score<=80))&
          ((df.reading_score>=75)&(df.reading_score<=95))&
          ((df.writing_score>=60)&(df.writing_score<=100))]


# In[24]:


new_df1


# In[ ]:




