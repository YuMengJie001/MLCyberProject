#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import keras
import sys
import h5py
import warnings
import scipy
warnings.filterwarnings("ignore")


# In[2]:


import STRIP 


# In[3]:


#You need to change this, according to your actual file location
filePath = "E:/360MoveData/Users/11813/Desktop/NYU2021Fall/MLforCyber/project/"


# In[67]:


#You need to change this, according to your actual entropy file location
entropy_file_name = filePath + "entropy_clean_sunglasses.h5"


# In[5]:


#load data and model, You need to change this, according to your actual situation
clean_validation_data = filePath + 'clean_validation_data.h5'
clean_test_data = filePath + 'clean_test_data.h5'
poisoned_test_data = filePath + 'sunglasses_poisoned_data.h5'
modelName = filePath +  'bd_net.h5'


# In[6]:


model = keras.models.load_model(modelName)


# In[68]:


entropy_clean_data = h5py.File(entropy_file_name1, "r")
entropy_clean = np.asarray(entropy_clean_data["data"])
entropy = [num/50 for num in entropy_clean1]
threshold = STRIP.cal_threshold(entropy)


# In[69]:


clean_x,clean_y = STRIP.data_load(clean_validation_data)
test_c_x,test_c_y = STRIP.data_load(clean_test_data)
test_p_x,test_p_y = STRIP.data_load(poisoned_test_data)


# In[73]:


clean_predict = np.argmax(model.predict(clean_x), axis=1)
clean_accuracy = np.mean(np.equal(clean_predict, clean_y))*100
print('Clean Date Classification Accuracy is :', clean_accuracy)


# In[75]:


G_model = STRIP.G(model)


# In[ ]:


test_predict= np.argmax(model.predict(test_c_x), axis=1)
poison_predict = np.argmax(model.predict(test_p_x), axis=1)


# In[83]:


pre01 = G_model.new_predict(test_x,clean_x, model,threshold,test_predict,10)
clean_test_accuracy = np.mean(np.equal(pre01, test_c_y))*100
print('Clean Date Classification Accuracy is :', clean_test_accuracy)


# In[ ]:


pre02 = G_model.new_predict(test_p_x,clean_x, model,threshold,poison_predict,10)
asr = np.mean(np.equal(pre02, test_y))*100
print('Attack Success Rate is :', asr)

