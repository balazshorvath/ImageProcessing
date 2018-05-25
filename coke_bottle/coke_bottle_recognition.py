from sklearn import tree
from helpers.nn_helper import *
import graphviz

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
"""
This is gone.

BOTTLE_ACCEPTABLE = 0
BOTTLE_NO_CAP = 1
#STICKER problems
BOTTLE_STICKER = 2
BOTTLE_NO_STICKER = 4
BOTTLE_STICKER_LEANING = 8
# FLUID problems
BOTTLE_FLUID = 16
BOTTLE_TOO_MUCH_FLUID = 32
BOTTLE_NOT_ENOUGH_FLUID = 64
BOTTLE_MISSING = 128
"""

features = load_features("data/features.csv")
fluid_features = {}
cap_label_features = {}

for k, v in features.items():
    fluid_features[k] = [v[len(v) - 1]]
    cap_label_features[k] = v[:len(v) - 1]

fluid_set = SampleSet(load_classification("data/classification_fluid.csv"), fluid_features, one_hot=False)
cap_and_label_set = SampleSet(load_classification("data/classification_cap_label.csv"), cap_label_features,
                              one_hot=False)

cap_label_results = []
fluid_results = []
for dzs in range(20):
    fluid_training, fluid_test = fluid_set.get_data_containers(0.4, randomize=True)
    cap_label_training, cap_label_test = cap_and_label_set.get_data_containers(0.4, randomize=True)

    fluid_training_all = fluid_training.get_once()
    cap_label_training_all = cap_label_training.get_once()
    """
        Cap label
    """
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(cap_label_training_all[0], cap_label_training_all[1])
    tree.export_graphviz(classifier,
                         out_file="cap_label_decision_tree.dot",
                         feature_names=["rect_area1", "rect_center_x1", "rect_center_y1", "rect_area2",
                                        "rect_center_x2", "rect_center_y2"],
                         class_names=["OK", "No bottle", "Cap", "Label", "Label and cap"])

    graphviz.render("dot", "png", "cap_label_decision_tree.dot")
    good = 0
    cap_label_test_all = cap_label_test.get_once()
    test_len = len(cap_label_test_all[0])
    for i in range(test_len):
        prediction = classifier.predict([cap_label_test_all[0][i]])
        actual_class = cap_label_test_all[1][i]
        if prediction == actual_class:
            good += 1
        # print("Prediction:\t\t%d " % prediction)
        # print("Actual class:\t%d " % actual_class)
    accuracy = good / test_len
    print(
        "%d out of %d was good.\nAccuracy was %.2f" % (good, test_len, accuracy))
    cap_label_results.append(accuracy)
    """
        Fluid
    """
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(fluid_training_all[0], fluid_training_all[1])
    tree.export_graphviz(classifier,
                         out_file="fluid_decision_tree.dot",
                         feature_names=["y_coord"],
                         class_names=["OK", "Low", "High"])

    graphviz.render("dot", "png", "fluid_decision_tree.dot")
    good = 0
    fluid_test_all = fluid_test.get_once()
    test_len = len(fluid_test_all[0])
    for i in range(test_len):
        prediction = classifier.predict([fluid_test_all[0][i]])
        actual_class = fluid_test_all[1][i]
        if prediction == actual_class:
            good += 1
        # print("Prediction:\t\t%d " % prediction)
        # print("Actual class:\t%d " % actual_class)
    accuracy = good / test_len
    print("%d out of %d was good.\nAccuracy was %.2f" % (good, test_len, accuracy))
    fluid_results.append(accuracy)
print("CP: " + str(cap_label_results))
print("Fluid: " + str(fluid_results))
