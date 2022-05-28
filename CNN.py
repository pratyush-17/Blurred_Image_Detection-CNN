#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pickle


# In[2]:


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD


# In[3]:



input_size = (96, 96)


# In[4]:



with open('X_train.pkl', 'rb') as picklefile:
    X_train = pickle.load( picklefile)


# In[5]:



with open('y_train.pkl', 'rb') as picklefile:
    y_train = pickle.load( picklefile)


# In[6]:




with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)


# In[7]:


with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)


# In[8]:


model = Sequential()


# In[9]:


model.add(Convolution2D(32, (5, 5), input_shape=(input_size[0], input_size[1], 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Convolution2D(64, (5, 5), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[11]:



model.add(Flatten())


# In[12]:



model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))


# In[13]:


model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))


# In[14]:


model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))


# In[15]:


model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[16]:


traindata = np.stack(X_train)
testdata = np.stack(X_test)
trainlabel = to_categorical(y_train)
testlabel = to_categorical(y_test)


# In[ ]:


model.fit(traindata, trainlabel, batch_size=128, epochs=25, verbose=1)
print("Model training complete...")
(loss, accuracy) = model.evaluate(testdata, testlabel, batch_size = 128, verbose = 1)
print("accuracy: {:.2f}%".format(accuracy * 100))


# In[18]:


print(model.summary())


# In[ ]:




