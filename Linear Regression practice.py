#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df = pd.read_csv("USA_Housing.csv")


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.columns


# In[16]:


df.head(10)


# In[17]:


#simple plots to check out the data
sns.pairplot(df)


# In[18]:


sns.distplot(df['Price'], bins = 20)


# In[19]:


sns.heatmap(df.corr(), annot = True)


# In[27]:


df.columns


# In[28]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[29]:


y = df['Price']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[32]:


X_train.head()
X_test.shape


# In[33]:


y_test


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


lm = LinearRegression()


# In[36]:


lm.fit(X_train, y_train)


# In[37]:


print(lm.intercept_)


# In[49]:


lm.coef_


# In[39]:


X_train.columns


# In[46]:


#create a df based on the coefficients
cdf = pd.DataFrame(lm.coef_ , X_train.columns , columns = ['Coeff'])
cdf
#just a 2D rep of the coefficient for each model predictor variable


# In[47]:


cdf.head()


# In[51]:


from sklearn.datasets import load_boston


# In[52]:


boston = load_boston()
boston.keys()


# In[53]:


predictions= lm.predict(X_test)


# In[54]:


predictions[0:10]


# In[58]:


plt.scatter(y_test, predictions)


# In[59]:


sns.distplot(y_test - predictions)


# In[62]:


from sklearn import metrics


# In[63]:


metrics.mean_absolute_error(y_test, predictions)


# In[64]:


metrics.mean_squared_error(y_test, predictions)


# In[65]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




