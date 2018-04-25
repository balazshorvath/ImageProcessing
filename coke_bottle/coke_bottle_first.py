# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:18:25 2018

@author: bala9
"""

import cv2
import numpy as np
from matplotlib import pylab as pl

image_file = "images\\cokes\\image005.jpg"

pl.figure()
image = cv2.imread(image_file)

pl.subplot(3, 2, 1)
pl.imshow(image)
image = np.array(image)
image = image[:,115:243]
pl.subplot(3, 1, 2)
pl.imshow(image)

# Blur and erode
#image = cv2.GaussianBlur(image, (15,15), 100)
kernel = np.ones((5,5),np.uint8)
image = cv2.erode(image, kernel, iterations=7)
pl.subplot(3, 2, 2)
pl.imshow(image)

imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 130, 255, 0)


im2, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
)
# Reload original image
image = cv2.imread(image_file)
image = np.array(image)
image = image[:,115:243]

# 

boxes = []
for cont in contours:    
    epsilon = 0.03*cv2.arcLength(cont,True)
    boxes.append(cv2.approxPolyDP(cont,epsilon,True))
print(boxes)
polygons = []
for box in boxes:
    polygon = []
    for boxi in box:
        polygon.append(boxi[0])
    polygons.append(np.array(polygon))

image = cv2.drawContours(image,boxes,-1,(255,0,0),thickness=cv2.FILLED)

print(polygons)

for polygon in polygons:
    polygon = polygon.reshape((-1,1,2))
    image = cv2.polylines(image, polygon, True, (0,0,0))

pl.subplot(3, 2, 4)
pl.imshow(image)
pl.show()





















