import glob
import os

import cv2
import numpy as np
from math import atan, sqrt


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


def process_image(image_file):
    """
        Crop the image to contian only the bottle in the middle.
        Show the image.
    """
    image = load_and_crop_image(image_file)

    """
        Use erosion on the cropped image.
        Show..
    """
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=7)

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
        Draw approximate polygons based on the contours and fill them with black.
        Show the new image.
    """
    boxes = []
    for cont in contours:
        epsilon = 0.02 * cv2.arcLength(cont, True)
        boxes.append(cv2.approxPolyDP(cont, epsilon, True))

    image = cv2.drawContours(image, boxes, -1, (0, 0, 0), thickness=cv2.FILLED)

    """
        Re-do previous steps, but with different erosion, threshold 
        and an additional blur effect to find the cap and the sticker.
    """
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=10)
    image = cv2.dilate(image, kernel, iterations=5)

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    thresh = cv2.blur(thresh, (35, 35))

    im2, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    """
        If there's still more, than two contours, try smoothing the colors even more.
    """
    if len(contours) > 2:
        thresh = cv2.blur(thresh, (15, 15))
        im2, contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    """
        Define and filter boxes. Calculate the size and the angle of the rectangles.
    """
    boxes = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        boxes.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))

    print(len(boxes))
    print(boxes)

    data = []
    for box in boxes:
        info = get_rectangle_info(box)
        if info[0] > 3000:
            data.append(info)
    if len(data) == 0:
        data.append((0.0, 0.0, 0.0, 0.0))
        data.append((0.0, 0.0, 0.0, 0.0))
    elif len(data) == 1:
        data.append((0.0, 0.0, 0.0, 0.0))
        if data[0][3] < 150:
            data = data[::-1]
    elif len(data) == 2:
        if data[0][3] < 150:
            data = data[::-1]
    elif len(data) > 2:
        print("The image %s has more, than two features! The first two rectangles are assumed to be the right ones."
              % file)
        data = data[0:2]
        if data[0][3] < 150:
            data = data[::-1]

    return data


if not os.path.exists("output\\coke_bottles"):
    os.makedirs("output\\coke_bottles")

data_file = open("output\\coke_bottles\\data.csv", "w")

for j, file in enumerate(glob.glob("images\\cokes\\*")):
    image_data = process_image(file)
    data_file.write("%s,%d" % (file, len(image_data)))
    data_file.write(",%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
        # image_data[0][0],  # area of the sticker rectangle
        image_data[0][2],  # center_x
        image_data[0][3],  # center_y
        image_data[1][0],  # area of the cap rectangle
        image_data[1][2],  # center_x
        image_data[1][3],  # center_y
    ))

data_file.close()
