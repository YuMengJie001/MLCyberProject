
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import keras
import sys
import h5py
import warnings
import math
import random
import time
import scipy
import keras
import cv2
import scipy






def data_load(filepath):
    data = h5py.File(filepath, 'r')
    x = np.array(data['data'])
    y = np.array(data['label'])
    x = x.transpose((0,2,3,1))
    return x, y


# Plot clean data picture
def plot_img(x,y):
    figure = plt.figure(figsize=(6,8))
    n, m = 3,3
    for i in range(1, n*m+1):
        index = np.random.randint(x.shape[0], size=1)
        img, label = (x[index], y[index])
        figure.add_subplot(n, m, i)
        plt.title("Label: {}".format(label))
        plt.axis("off")
        plt.imshow(img[0]/255)
    plt.show()


def overlap(background, overlay):
    new_image = cv2.addWeighted(background,1.5,overlay,1,0,dtype = cv2.CV_32F)
    return (new_image.reshape(55,47,3))




def entropy_calculate(background,Overlay_data,n,model):
    overlaped_imgs = [0] * n
    indexs = np.random.randint(0,len(Overlay_data), size=n)
    for x in range(n):
        overlaped_imgs[x] = overlap(background, Overlay_data[indexs[x]])
    predicts = model.predict(np.array(overlaped_imgs))
    EntropySum = -np.nansum(predicts*np.log2(predicts))
    return EntropySum


#background_data: background  input_data:img that overlaped to background  N:sample that overlap to one background img  cal_num:the total img that
# need to overlaped    model:model we had
def cal_entropy_all(background_data,Overlay_data,N, cal_num,model):
    entropy_sum = [0] * cal_num
    for j in range(cal_num):
        
        x_background = background_data[j*2] 
        entropy_sum[j] = entropy_calculate(x_background,Overlay_data, N, model)
    entropy_list = [x  for x in entropy_sum] # get entropy for 2000 clean inputs
    return entropy_list

def cal_threshold(entropy_list):
    (mu, sigma) = scipy.stats.norm.fit(entropy_list)
    threshold = scipy.stats.norm.ppf(0.05, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be
    return threshold



class G(keras.Model):
    def __init__(self, model):
        super(G, self).__init__()
        self.model = model

    def new_predict(self,input_img,Overlay_data, model,threshold,predict_lable,N):
        length = len(input_img)
        predicts = np.zeros(length)
        for i in range(length):
            now_entropy = entropy_calculate(input_img[i],Overlay_data, N, model)
            if now_entropy < threshold:
                predicts[i] = max(predict_lable) + 1
            else:
                predicts[i] = predict_lable[i]
        return predicts


