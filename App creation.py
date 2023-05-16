#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
filename = pd.read_csv("https://raw.githubusercontent.com/LinkedInLearning/dsm-bank-model-2870047/main/bankData/bank.csv ")


# In[5]:


filename.(head)


# In[6]:


filename.head()


# In[7]:


from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[9]:


df = pd.read_csv("https://raw.githubusercontent.com/LinkedInLearning/dsm-bank-model-2870047/main/bankData/bank.csv", header = None)


# In[10]:


df.head()


# #drop campaign related columns

# In[11]:


#drop campaign related columns
df.drop(df.iloc[:, 8:16], inplace = True, axis = 1)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[12]:


df.head()


# In[14]:


numeric_data = df.iloc[:, [0, 5]].values
numeric_df = pd.DataFrame(numeric_data, dtype = object)
numeric_df.columns = ['age', 'balance']


# In[15]:


age_std_scale = StandardScaler()
numeric_df['age'] = age_std_scale.fit_transform(numeric_df[['age']])


# In[16]:


balance_std_scale = StandardScaler()
numeric_df['balance'] = balance_std_scale.fit_transform(numeric_df[['balance']])


# In[17]:


X_categoric = df.iloc[:, [1,2,3,4,6,7]].values


# In[24]:


one = OneHotEncoder()
categoric_data = one.fit_transform(X_categoric).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = one.get_feature_names()


# In[25]:


categoric_df.head()


# In[26]:


numeric_data = df.iloc[:, [0, 5]].values
numeric_df = pd.DataFrame(numeric_data, dtype = object)
numeric_df.columns = ['age', 'balance']


# In[27]:


age_std_scale = StandardScaler()
numeric_df['age'] = age_std_scale.fit_transform(numeric_df[['age']])


# In[28]:


balance_std_scale = StandardScaler()
numeric_df['balance'] = balance_std_scale.fit_transform(numeric_df[['balance']])


# In[29]:


numeric_df.head()


# In[34]:


X_final = pd.concat([numeric_df, categoric_df], axis = 1)


# In[35]:


x_final.head()


# In[36]:


ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X_categoric).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names()


# In[37]:


X_final = pd.concat([numeric_df, categoric_df], axis = 1)


# In[40]:


X_final.head()


# In[39]:


X_final.head()


# #train model
# 

# In[42]:


rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)


# In[48]:


from sklearn.metrics import confusion_metrix
confusin_metrix(y_test,y_pred)


# In[49]:


rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_final,y, test_size= 0.2)

rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)


# In[61]:


y_pred=rfc.predict(X_test)
from sklearn.metrics import confusion_matrix
confusin_matrix=(y_test,y_pred)


# In[62]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




