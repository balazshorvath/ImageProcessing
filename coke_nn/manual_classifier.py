import cv2
import glob

from matplotlib import pyplot as pl

out_file = open("data\\classification.csv", "w")

for j, file in enumerate(glob.glob("images\\*")):
    image = cv2.imread(file)
    pl.figure()
    pl.imshow(image)
    pl.show()
    image_class = input("What was the class of the image? ").strip()
    out_file.write("%s;%s\n" % (file, image_class))
    out_file.flush()

out_file.close()
