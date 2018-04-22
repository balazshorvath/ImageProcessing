import glob
import os

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pylab import *
from scipy.ndimage import filters

if not os.path.exists("output\\samples"):
    os.makedirs("output\\samples")

masks = [
    np.matrix([
        [-1., -1., 0.],
        [-1., 0., 1.],
        [0., 1., 1.]
    ]),
    np.matrix([
        [0., -1., -1.],
        [1., 0., -1.],
        [1., 1., 1.]
    ]),
    np.matrix([
        [0., -1., 0.],
        [-1., 4., -1.],
        [0., -1., 0.]
    ]),
    np.matrix([
        [-1., -1., -1.],
        [-1., 8., -1.],
        [-1., -1., -1.]
    ])
]

data_file = open("output\\samples\\data.txt", "w")

for j, file in enumerate(glob.glob("images\\patterns\\*")):
    print(file)
    img = Image.open(file)
    if img.mode != "L":
        img = img.convert("L")

    gray_scale = np.int16(np.array(img))
    data_file.write("\n" + file + ":\n" + str(gray_scale))
    filename = file[file.rfind("\\") + 1:]
    plt.figure(j)
    plt.subplot(5, 1, 1)
    imshow(gray_scale)
    for i, mask in enumerate(masks):
        result = np.clip((filters.convolve(gray_scale, mask)), 0, 255)
        data_file.write("\nMask " + str(i) + "\n" + str(result))
        plt.subplot(5, 1, i + 2)
        imshow(result)  # , cmap = "gray")
        result_image = Image.fromarray(result, mode="L")
        result_image.save("output\\samples\\mask" + str(i) + "_" + filename)
    plt.savefig("output\\samples\\result" + str(j) + filename + ".png")
    # plt.show()
