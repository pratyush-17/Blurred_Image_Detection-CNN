#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pickle


# In[2]:


from keras.preprocessing import image


# In[3]:


input_size = (96, 96)
X_test = []
y_test = []


# In[4]:


dgbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')


# In[5]:


dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})


# In[6]:


nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())


# In[7]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'


# In[8]:


for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')
print("Testset: Artificially Blurred loaded...")


# In[9]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'


# In[10]:


for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')

print("Trainset: Naturally Blurred loaded...")


# In[11]:


with open('X_test.pkl', 'wb') as picklefile:
    pickle.dump(X_test, picklefile)


# In[12]:


with open('y_test.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)


# In[ ]:




