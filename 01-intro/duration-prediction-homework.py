#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Set default display format for pandas.

# In[2]:


pd.set_option('display.float_format', '{:.2f}'.format)


# Read function for dataframes. Including homework answears Q1, Q2, Q3.

# In[3]:


def read_dataframe(url, month):
    df = pd.read_parquet(url)
    
    print(f"Column count for month {month} is {len(df.columns.tolist())}.")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    print(f"The standard deviation for month {month} is {df.duration.std()}")
    
    print(df.duration.describe(percentiles=[0.90, 0.92, 0.95, 0.98, 0.99, 0.995]))

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].astype(str)
    
    return df


# Read dataframes from Yellow trip data 2023 January and February.

# In[4]:


df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet', "January")
df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet', "February")


# One-hot encode and DictVectorizer init. Get feaute matrixes.

# In[5]:


categorical = ['PULocationID', 'DOLocationID']
dv = DictVectorizer()


# In[6]:


train_dicts = df_train[categorical].to_dict(orient = 'records')
X_train = dv.fit_transform(train_dicts)


# In[7]:


val_dicts = df_val[categorical].to_dict(orient = 'records')
X_val = dv.transform(val_dicts)


# Get the dimensionality of X_train matrix . Including homework answear Q4.

# In[8]:


X_train.shape


# In[9]:


target = 'duration'

Y_train = df_train[target].values
Y_val = df_val[target].values


# Training the model with duration.

# In[10]:


lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_pred_train = lr.predict(X_train)
Y_pred_val = lr.predict(X_val)


# RMSE on train. Including homework answear Q5.

# In[11]:


mean_squared_error(Y_train, Y_pred_train, squared = False)


# RMSE on validation. Including homework answear Q6.

# In[12]:


mean_squared_error(Y_val, Y_pred_val, squared = False)


# Save the homework model.

# In[13]:


with open('models/lin_reg_homework.bin', 'wb') as f_out:
    pickle.dump((dv,lr), f_out)


# In[ ]:




