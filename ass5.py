#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Social_Network_ads.csv')
df


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


x=df.iloc[:,[2,3]].values


# In[10]:


x


# In[11]:


y=df.iloc[:,4].values
y


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[13]:


x_train


# In[14]:


x_train.shape


# In[15]:


y_train


# In[16]:


y_train.shape


# In[17]:


x_test


# In[18]:


x_test.shape


# In[19]:


y_test


# In[20]:


y_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


sc_x=StandardScaler()


# In[23]:


x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)


# In[24]:


x_train


# In[25]:


x_test


# In[26]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[27]:


y_pred=classifier.predict(x_test)


# In[28]:


y_pred


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[31]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)*100
accuracy


# In[32]:


tp = cm[0,[0]]
print('True Positive :',tp)


# In[33]:


fp = cm[0,[1]]
print('False Positive :',fp)


# In[35]:


fn = cm[1,[0]]
print('False Negative :',fn)


# In[36]:


tn = cm[1,[1]]
print('True Negative :',tn)


# In[37]:


accuracy_cm = ((tp+tn)/(tp+fp+fn+tn))
print('Accuracy: ',accuracy_cm*100)


# In[38]:


error_rate_cm = ((fp+fn)/(tp+fp+fn+tn))
print('Error Rate: ',error_rate_cm*100)


# In[39]:


precision_cm = (tp/(tp+fp))
print('Precision : ',precision_cm*100)


# In[40]:


recall_cm =(tp/(tp+fn))
print('Sensitivity :',recall_cm*100)


# In[41]:


specificity_cm = (tn/(tn+fp))
print('Specificity :',specificity_cm*100)

