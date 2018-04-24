from sklearn import tree
import csv

"""
    Bottle description:
        file_path,state
    state = PROBLEM1|PROBLEM2
        if PROBLEM1 and PROBLEM2 is present.

    state = 0
        if there's no problem

    Bottle features:
        file_path,rect_count
        rect_area1,rect_center_x1,rect_center_y1
        rect_area2,rect_center_x2,rect_center_y2
        ...
        rect_arean,rect_center_xn,rect_center_yn


    Features:
        [rect_count, rect_area1, rect_center_x1, rect_center_y1, rect_area2, rect_center_x2, rect_center_y2]
"""
BOTTLE_ACCEPTABLE = 0
BOTTLE_NO_CAP = 1
"""
    STICKER problems
"""
BOTTLE_STICKER = 2
BOTTLE_NO_STICKER = 4
BOTTLE_STICKER_LEANING = 8
"""
    FLUID problems
"""
BOTTLE_FLUID = 16
BOTTLE_TOO_MUCH_FLUID = 32
BOTTLE_NOT_ENOUGH_FLUID = 64

BOTTLE_MISSING = 128


def load_classification(file_path):
    return []


def load_features(file_path):
    csv_file = open(file_path, "r")
    features = {}
    for row in csv_file:
        r = row.split(",")
        parsed_data = []
        for d in r:
            parsed_data.append(float(d.strip()))
        features[r[0]] = parsed_data
    return []


dt = tree.DecisionTreeClassifier()
