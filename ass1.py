#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('train.csv')
df


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.size


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df


# In[11]:


df["cabin"]=df["cabin"].replace(to_replace=np.nan, value="unknown")
df


# In[12]:


df["age"]=df["age"].interpolate()


# In[13]:


df


# In[14]:


df.isnull().sum()


# In[17]:


df["embarked"]=df["embarked"].replace(to_replace=np.nan, value="unknown")
df


# In[18]:


df.isnull().sum()


# In[19]:


df.dtypes


# In[20]:


quantitative_data=pd.get_dummies(df.embarked,prefix='embarked')


# In[21]:


quantitative_data


# In[22]:


df=df.join(quantitative_data)
df


# In[25]:


df.drop(['embarked'],axis=1,inplace=True)


# In[26]:


df


# In[28]:


quantitative_sex = pd.get_dummies(df.sex,prefix = 'sex')
quantitative_sex


# In[29]:


df = df.join(quantitative_sex)
df


# In[30]:


df.drop(['sex'],axis =1,inplace=True)
df


# In[ ]:




