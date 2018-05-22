import os
import glob
import cv2
import numpy as np

current_id = 0
out_dir = "images/split/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in glob.glob("images/*.jpg"):
    img = np.array(cv2.imread(file))
    cv2.imwrite("%s%04d.jpg" % (out_dir, current_id), img[:, :115])  # out_dir + str(current_id) + ".jpg", img[:, :115])
    current_id += 1
    cv2.imwrite("%s%04d.jpg" % (out_dir, current_id), img[:, 115:243])
    current_id += 1
    cv2.imwrite("%s%04d.jpg" % (out_dir, current_id), img[:, 243:])
    current_id += 1

"""
    return img[:, 115:243]
"""
