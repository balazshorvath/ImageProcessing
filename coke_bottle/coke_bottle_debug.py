import cv2
import numpy as np
from matplotlib import pylab as pl
from math import atan, sqrt

image_file = "images\\cokes\\image008.jpg"


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
    rect_data = ((len1 * len2), atan(len_line_y))
    print("area %.2f, angle %.2f" % rect_data)
    return rect_data


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
# image = cv2.GaussianBlur(image, (15,15), 100)
kernel = np.ones((5, 5), np.uint8)
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

image = cv2.drawContours(image, boxes, -1, (0, 0, 0), thickness=cv2.FILLED)

pl.subplot(3, 3, 4)
pl.imshow(image)

"""
    Re-do previous steps, but with different erosion, threshold 
    and an additional blur effect to find the cap and the sticker.
"""
kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(image, kernel, iterations=10)
image = cv2.dilate(image, kernel, iterations=5)
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=10)

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
    if there's still more, than two contours, try smoothing the colors even more.
"""
if len(contours) > 2:
    thresh = cv2.blur(thresh, (15, 15))

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
    x, y, w, h = cv2.boundingRect(cont)
    boxes.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    # rect = cv2.minAreaRect(cont)
    # box = cv2.boxPoints(rect)
    # boxes.append(np.int0(box))

image_contours = cv2.drawContours(image, boxes, -1, (255, 255, 255), thickness=cv2.FILLED)
print(len(boxes))
print(boxes)

pl.subplot(3, 3, 7)
pl.imshow(image_contours)

data = []
for box in boxes:
    info = get_rectangle_info(box)
    if info[0] > 3000:
        data.append(info)

pl.show()
