from sklearn import tree
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
    csv_file = open(file_path, "r")
    labels = {}
    for row in csv_file:
        r = row.split(",")
        labels[r[0].strip()] = float(r[1].strip())
    return labels


def load_features(file_path):
    csv_file = open(file_path, "r")
    features = {}
    for row in csv_file:
        r = row.split(",")
        parsed_data = []
        for d in r[1:]:
            parsed_data.append(float(d.strip()))
        features[r[0]] = parsed_data
    return features


bottle_features = load_features("data.csv")
bottle_labels = load_classification("simple_classification.csv")

training_features = []
training_labels = []
for i, feature_key in enumerate(bottle_labels):
    training_features.append(bottle_features[feature_key])
    training_labels.append(bottle_labels[feature_key])

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(training_features, training_labels)
tree.export_graphviz(classifier,
                     out_file="decision_tree.dot",
                     feature_names=["rect_count", "rect_area1", "rect_center_x1", "rect_center_y1", "rect_area2",
                                    "rect_center_x2", "rect_center_y2"],
                     class_names=["Perfect", "Missing cap", "Missing or bad label", "No bottle"])
graphviz.render("dot", "png", "decision_tree.dot")
test_labels = load_classification("simple_test_classification.csv")

for feature_key in test_labels:
    print("File: %s" + feature_key)
    prediction = classifier.predict([bottle_features[feature_key]])
    actual_class = test_labels[feature_key]
    print("Prediction:\t\t%d " % prediction)
    print("Actual class:\t%d " % actual_class)