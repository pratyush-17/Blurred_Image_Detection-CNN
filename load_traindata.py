#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image


# In[2]:


from keras.preprocessing import image


# In[3]:


X_train = []
y_train = []

input_size = (96, 96)


# In[8]:


folderpath ="CERTH_ImageBlurDataset/TrainingSet/Undistorted/"


# In[9]:


for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size = input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(0)
    else:
        print(filename, 'not a pic')
print("Trainset: Undistorted loaded...")


# In[11]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'


# In[12]:


for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Artificially Blurred loaded...")


# In[13]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'


# In[14]:


for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Naturally Blurred loaded...")


# In[15]:


with open('X_train.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)


# In[16]:



with open('y_train.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)


# In[ ]:




