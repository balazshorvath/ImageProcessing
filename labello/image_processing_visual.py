from sys import argv
from matplotlib import pylab as pl

import cv2
import numpy as np


def blur_image(i):
    i = cv2.blur(i, (31, 31))
    i = cv2.morphologyEx(i, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8), iterations=2)
    return i


_img = cv2.imread(argv[1])
pl.figure()
pl.subplot(3, 3, 1)
pl.imshow(_img)
img = np.array(_img, copy=True)
# TODO
img = blur_image(img)
img = blur_image(img)
#
pl.subplot(3, 3, 2)
pl.imshow(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img, 20, 255, 0)
# thresh = cv2.adaptiveThreshold(
#     img,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,
#     2
# )
pl.subplot(3, 3, 3)
pl.imshow(img)

_, contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)
img = np.array(_img, copy=True)
img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
pl.subplot(3, 3, 4)
pl.imshow(img)

# Let's just assume, that contours will contain only one element,
# or that the first will be the go-to contours
x, y, w, h = cv2.boundingRect(contours[0])
img = np.array(_img[y: y + h, x: x + w], copy=True)
# Area, avg red, avg green, avg blue
features = ([w * h, img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()])

pl.subplot(3, 3, 5)
pl.imshow(img)

print(features)
pl.show()
