#import os.args
import cv2
import numpy as np
from matplotlib import pylab as pl

image_file = "images\\cokes\\image019.jpg"


def load_and_crop_image(file):
    image = cv2.imread(file)
    image = np.array(image)
    return image[:,115:243]


"""
    Load and show the original image.
"""
pl.figure()
image = cv2.imread(image_file)
pl.subplot(3, 3, 1)
pl.imshow(image)
"""
    Crop the image to contian only the bottle in the middle.
    Show the image.
"""
image = load_and_crop_image(image_file)
pl.subplot(3, 3, 2)
pl.imshow(image)

"""
    Use erosion on the cropped image.
    Show..
"""
#image = cv2.GaussianBlur(image, (15,15), 100)
kernel = np.ones((5,5),np.uint8)
image = cv2.erode(image, kernel, iterations=7)
pl.subplot(3, 3, 3)
pl.imshow(image)

"""
    Convert to gray-scale and then binary.
"""
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 130, 255, 0)

"""
    Find the contours.
"""
im2, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
)

"""
    Reload original image and crop it, OpenCV alters the original.
"""
image = load_and_crop_image(image_file)

"""
    Draw approximate poligons based on the contours and fill them with black.
    Show the new image.
"""
boxes = []
for cont in contours:
    epsilon = 0.02 * cv2.arcLength(cont, True)
    boxes.append(cv2.approxPolyDP(cont, epsilon, True))

image= cv2.drawContours(image,boxes,-1,(0,0,0),thickness=cv2.FILLED)

pl.subplot(3, 3, 4)
pl.imshow(image)

"""
    Re-do previous steps, but with different erosion, threshold 
    and an additional blur effect to find the cap and the sticker.
"""
kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(image, kernel, iterations=10)
pl.subplot(3, 3, 5)
pl.imshow(image)

imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)
thresh = cv2.blur(thresh, (35, 35))

im2, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
)
"""
    Reload original image and draw the contours around the cap and sticker.
"""
image = load_and_crop_image(image_file)
image_contours = cv2.drawContours(image, contours, -1, 255, 3)
pl.subplot(3, 3, 6)
pl.imshow(image_contours)

"""
    Reload original image and draw the rectangles around the cap and sticker.
"""
image = load_and_crop_image(image_file)
boxes = []
for cont in contours:    
    rect = cv2.minAreaRect(cont)
    box = cv2.boxPoints(rect)
    boxes.append(np.int0(box))

image_contours = cv2.drawContours(image, boxes, -1, 255, 3)
pl.subplot(3, 3, 7)
pl.imshow(image_contours)
pl.show()






