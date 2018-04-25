import glob

import cv2
import numpy as np


# from matplotlib import pylab as pl


def blur_image(img, kernel):
    img = cv2.blur(img, (31, 31))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    return img


def collect_image_data(directory_pattern):
    kernel = np.ones((15, 15), np.uint8)
    features = []
    for j, file in enumerate(glob.glob(directory_pattern)):
        _img = cv2.imread(file)
        img = np.array(_img, copy=True)
        # TODO 
        img = blur_image(img, kernel)
        img = blur_image(img, kernel)
        #

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 20, 255, 0)
        _, contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Let's just assume, that contours will contain only one element,
        # or that the first will be the go-to contours
        x, y, w, h = cv2.boundingRect(contours[0])
        img = np.array(_img[y: y + h, x: x + w], copy=True)
        # Area, avg red, avg green, avg blue
        features.append([w * h, img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()])

    return features
