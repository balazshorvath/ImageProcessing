import cv2
import numpy as np
from math import atan, sqrt
from matplotlib import pylab as pl


def load_and_crop_image(file):
    img = cv2.imread(file)
    img = np.array(img)
    return img[:, 115:243]


def get_rectangle_info(rct):
    """
    Rectangles are in a format of vertices in clockwise order.

    :param rct: rectangle
    :return: tuple(area, angle)
    """
    len_line_x = rct[0, 0] - rct[1, 0]
    len_line_y = rct[0, 1] - rct[1, 1]
    if len_line_x == 0:
        m1 = len_line_y
    else:
        m1 = len_line_y / len_line_x
    len1 = sqrt(len_line_y ** 2 + len_line_x ** 2)
    print("Side 1: len_line_x %.2f, len_line_y %.2f, m %.2f, len %.2f" % (len_line_x, len_line_y, m1, len1))
    len_line_x = rct[1, 0] - rct[2, 0]
    len_line_y = rct[1, 1] - rct[2, 1]
    if len_line_x == 0:
        m2 = len_line_y
    else:
        m2 = len_line_y / len_line_x
    len2 = sqrt(len_line_y ** 2 + len_line_x ** 2)
    print("Side 2: len_line_x %.2f, len_line_y %.2f, m %.2f, len %.2f" % (len_line_x, len_line_y, m2, len2))
    center_x = (rct[0, 0] + rct[2, 0]) / 2
    center_y = (rct[0, 1] + rct[2, 1]) / 2

    rect_data = ((len1 * len2), atan(len_line_y), center_x, center_y)
    print("area %.2f, angle %.2f, center_x %.2f, center_y %.2f" % rect_data)
    return rect_data


image_file = "images\\image046.jpg"
"""
    Crop the image to contian only the bottle in the middle.
    Show the image.
"""
image = load_and_crop_image(image_file)
pl.figure()
pl.subplot(2, 3, 1)
pl.imshow(image)

kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(image, kernel, iterations=5)
image = cv2.dilate(image, kernel, iterations=10)
pl.subplot(2, 3, 2)
pl.imshow(image)

image = cv2.blur(image, (25, 25))
pl.subplot(2, 3, 3)
pl.imshow(image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)
pl.subplot(2, 3, 4)
pl.imshow(thresh)

im2, contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)
boxes = []
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    boxes.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    print(y)

print(len(boxes))
print(boxes)

image = load_and_crop_image(image_file)
image = cv2.drawContours(image, boxes, -1, (0, 0, 0), thickness=cv2.FILLED)

pl.subplot(2, 3, 5)
pl.imshow(image)

pl.show()
