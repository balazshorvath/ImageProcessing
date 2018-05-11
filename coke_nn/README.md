# Notes for classifying coke bottles with DNN
Using Multi-layer Perceptron.

This a newer version of the "coke_bottle" task. I'll use the same technique to extract the features from the images.
Later this method should be extended with the fluid level features as well.

## Images
### Classes
Images may have the following classes:
 - (0, 0000) Bottle is OK
 - (1, 0001) Cap missing
 - (2, 0010) Label missing, or was put on wrong
 - (3, 0011) Label and cap
 - (4, 0100) Fluid level is not right
 - (5, 0101) Fluid and cap
 - (6, 0110) Fluid and label
 - (7, 0111) Bottle is not present
 - (8, 1000) Deformed bottle

The number in front of the classes are the integer values for them.

#### manual_classifier.py
Simple script for pre-processing the data.
Opens every file in a directory, after closing the window it will prompt for the class of the image.
The output is a csv file.
Example ({filename};{class}):
```
file000.png;1
file001.png;3
file002.png;1
file003.png;2
```
### Features
#### Label/Cap
Find the area on the image, where the label and the cap might be and draw an estimate rectangle around them.
Features are:
 - Area of the rectangle
 - x length
 - y length
#### Fluid level
This is not implemented yet.
## Image processing
### Step 1 - Crop
First of all there are 3 bottles on every image and only the one in the middle is the one to classify.
Fortunately the bottles are in the same place with little displacement.
### Step 2 - Find the cap and the label
#### Step 2.1 - Color the white areas black
This will help separate the light colored cap and label when applying a threshold next time.
 1. Erode the image
 2. Threshold with a high value
 3. Find contours
 4. Draw polygons, fill with black
#### Step 2.2 - Find the two boxes
 1. Erode to make the text on the label disappear. This had to be done several times.
 2. Dilate to somewhat restore the shape of the important areas. This has to be done lesser times, otherwise the contours start to appear.
 3. Threshold with lower value
 4. Blur with normalized box filter one, or two times to simply get rid of the small contours.
 5. Calculate bounding rectangles
 6. Filter the resulting rectangles based on their area. If they are too small, simply ignore them.
### Step 3 - Find the fluid level
This is not implemented yet!

