#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
boston=load_boston()


# In[2]:


print(boston)


# In[4]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[5]:


df_x=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[6]:


df_y=pd.DataFrame(boston.target)


# In[7]:


df_x


# In[8]:


df_y


# In[10]:


df_x.describe()


# In[11]:


df_y.describe()


# In[12]:


df_x.shape


# In[13]:


df_x.columns


# In[14]:


reg=linear_model.LinearRegression()


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.3)


# In[16]:


reg.fit(x_train,y_train)


# In[19]:


print(reg.coef_)


# In[20]:


y_pred=reg.predict(x_test)
print(y_pred)


# In[21]:


print(y_test)


# In[22]:


print(np.mean(y_pred-y_test)**2)


# In[24]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))

