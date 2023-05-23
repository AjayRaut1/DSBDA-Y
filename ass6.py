#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn as sns


# In[3]:


df=pd.read_csv('iris.csv')
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


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


sns.relplot(data = df, x='petal.length', y='petal.width', hue='variety',size=2.5)


# In[12]:


sns.relplot(data = df, x='sepal.length', y='sepal.width', hue='variety',size=2.5)


# In[13]:


sns.pairplot(df, hue = 'variety')


# In[14]:


plt.figure(figsize = (15, 10))
plt.subplot(2,2,1)
sns.boxplot(data = df, x='variety',y='petal.length')

plt.subplot(2,2,2)
sns.boxplot(data = df, x='variety',y='petal.width')

plt.subplot(2,2,3)
sns.boxplot(data = df, x='variety',y='sepal.length')

plt.subplot(2,2,4)
sns.boxplot(data = df, x='variety',y='sepal.width')


# In[15]:


sns.boxplot(data=df).set_title("Distribution of sepal.length, sepal.width, petal.length, petal.width for iris_setosa, iris-Versicolor and iris-Virginica ")


# In[16]:


df.corr()


# In[17]:


plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(),annot = True,fmt="f").set_title("Corelation of attributes")


# In[18]:


x = df.iloc[:,0:4].values
x


# In[19]:


y=df.iloc[:,4].values
y


# In[20]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
y


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[22]:


x_train


# In[23]:


x_train.shape


# In[24]:


x_test


# In[25]:


x_test.shape


# In[26]:


y_train


# In[27]:


y_train.shape


# In[28]:


y_test


# In[29]:


y_test.shape


# In[30]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train, y_train)


# In[31]:


prediction=model.predict(x_test)


# In[32]:


prediction


# In[33]:


y_test


# In[34]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction)
cm


# In[35]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,prediction)*100
accuracy


# In[36]:


from sklearn.metrics import precision_score
precision = precision_score(y_test,prediction,average='micro')*100
precision


# In[37]:


from sklearn.metrics import recall_score
recall = recall_score(y_test,prediction,average='micro')*100
recall

