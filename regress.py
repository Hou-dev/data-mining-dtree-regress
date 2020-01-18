#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[2]:


# reading the csv file
data = pd.read_csv('winequality-white.csv',sep=';')
data2 = pd.read_csv('Complex9_GN32.csv')
data.head()


# In[3]:


# set the names for dataframe
x = data[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',      
        'total sulfur dioxide','density','pH','sulphates','alcohol']]
#find the zscore and normalize
x.apply(zscore)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled)
# Leaving the qualiity as is
dataset['quality'] = data['quality']
dataset.rename(columns={'0':'fixed acidity', '1':'volatile acidity','2':'citric acid','3':'residual sugar',
                        '4':'chlorides','5':'free sulfur dioxide','6':'total sulfur dioxide',
                        '7':'density','8':'pH','9':'sulphates','10':'alcohol'})
# making a new class and change the name according to the value of quality
dataset.loc[dataset['quality'] == 8, 'class'] = 'A'
dataset.loc[dataset['quality'] == 9, 'class'] = 'A'
dataset.loc[dataset['quality'] == 8, 'class'] = 'A'
dataset.loc[dataset['quality'] == 7, 'class'] = 'B'
dataset.loc[dataset['quality'] == 6, 'class'] = 'C'
dataset.loc[dataset['quality'] == 5, 'class'] = 'D'
dataset.loc[dataset['quality'] == 4, 'class'] = 'E'


# In[15]:


# making the arrays to test functons
a = np.array([0,1,1,1,1,2,2,3]).reshape(-1,1)
b = np.array(['A','A','A','E','E','D','D','C'])

import scipy.stats
# finding entropy using scipy
def ent(data):
    data = data.value_counts()          
    entropy = scipy.stats.entropy(data) 
    return entropy


# In[45]:


def entropy(a,b):
    
    out1 = len(a)
    # finding z score and classify outliers
    z= np.abs(zscore(a))
    a = a[(z < 1).all(axis=1)]
    out2 = len(a)
    # kmeans clustering on the cleaned data
    kmeans = KMeans(n_clusters=2, random_state=0).fit(a)
    pred = kmeans.predict(a)
    s1 = pd.Series(pred)
    # finding the percentage outlier
    pcentout = 1 - (out2/out1)
    print(ent(s1),pcentout)
    
entropy(a,b)


# In[ ]:


def ordinal_variation(a,b):
    return 0


# In[80]:


def variance(a,b):
    # kmeans clustering
    kmean = KMeans(n_clusters=3,random_state=0).fit(a)
    pred = kmean.predict(a)
    # find the number of elements per cluster
    count = np.bincount(pred)
    var = 0
    variance = 0
    for cell in count:
        a = count
        b = len(pred)
        # finding the variance
        var += (np.var(pred))*(a/b)
    
    for cell in var:
        # summing array to find total variance
        variance = sum(var)
    return variance,print('No Outlier in KMeans')
variance(a,b)


# In[50]:


def mdist(d):
    from scipy.spatial.distance import pdist
    # finding the zscore
    zs= np.abs(zscore(d))
    # find the manhatan distance using scipy
    zs = pdist(zs,'seuclidean',V=None)
    return zs
mdist(a)


# In[52]:


# naming columns
x = data[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
        'density','pH','sulphates','alcohol']]
y = data['quality']
# train and test data split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle= True)
# use multiple linear regression
Reg = LinearRegression()
Reg.fit(X_train, y_train)
# find the coeffcients
print('Coef: ', Reg.coef_)


# In[7]:


# cleaner linear model using least squares
model = sm.OLS(y,x).fit()
print_model = model.summary()
print(print_model)


# In[ ]:




