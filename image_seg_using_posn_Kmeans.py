# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:26:34 2020

@author: vinayak
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

img_ = cv2.imread('image.jpg') 
img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
original_shape = img_.shape
print(img_.shape)

plt.imshow(img_) # as RGB Format
plt.show()
#Standardise image

img_dist = np.zeros(shape=(1067,1600,5))


for i in range(1067):
    for j in range(1600):
        for k in range(3):
            img_dist[i][j][k]=img_[i][j][k]
        img_dist[i][j][3] = (i/1067)*255
        img_dist[i][j][4] = (j/1600)*255

# Flattening of image
all_pixels = img_dist.reshape((-1,5))
print(all_pixels.shape)

from sklearn.cluster import KMeans
dominant_colors = 4
km = KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)

centers = km.cluster_centers_
centers = np.array(centers,dtype='uint8')

print(centers)

i = 1

plt.figure(0,figsize=(8,2))


colors = []

for each_col in centers:
    plt.subplot(1,8,i)
    plt.axis("off")
    i+=1
    
    colors.append(each_col[:3])
    
    #Color Swatch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col[:3]
    plt.imshow(a)
    
plt.show()

new_img_generated = np.zeros((1707200,3),dtype="uint8")
print(new_img_generated.shape)

for i in range(new_img_generated.shape[0]):
    new_img_generated[i] = colors[km.labels_[i]]

new_img_generated = new_img_generated.reshape((original_shape))
plt.imshow(new_img_generated)
plt.show()