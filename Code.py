#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn


# In[3]:


data = pd.read_csv('Downloads\datasets_293841_602591_top50.csv', encoding='ISO-8859-1')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[8]:


data.isnull.Sum()


# In[9]:


data['Artist.Name'].value_counts()[:10].plot(kind='bar')


# In[10]:


data[data['Artist.Name']=='Ed Sheeran']


# In[11]:


data['Genre'].value_counts().plot(kind='bar')


# In[12]:


data[data['Genre']=='dance pop']


# In[13]:


plt.figure(figsize=(10,8))
data['Genre'].value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[14]:


data['Popularity'].plot(kind='hist')


# In[15]:


data['Energy'].plot(kind='hist')


# In[16]:


plt.figure(figsize=(8,6))
plt.scatter(x= data['Popularity'],y=data['Energy'])
plt.xlabel('Popularity')
plt.ylabel('Energy')
plt.show()


# In[17]:


plt.figure(figsize=(8,6))
plt.scatter(x= data['Popularity'],y=data['Danceability'])
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.show()


# In[19]:


data1 = pd.read_csv('Downloads\datasets_293841_602591_top50.csv', encoding='ISO-8859-1')
data1.head()


# In[20]:


data1 =data1.drop(['Unnamed: 0','Track.Name','Artist.Name'],axis=1)
x = data1.drop(['Popularity'],axis=1)
y = data1['Popularity']


# In[21]:


x= pd.get_dummies(x)


# In[22]:


x.head()


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size=0.3)


# In[26]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[25]:





# In[ ]:




