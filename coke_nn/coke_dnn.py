from helpers.nn_helper import *
import tensorflow as tf

"""
CSV parsing
"""


def load_classification(file_path):
    csv_file = open(file_path, "r")
    _labels = {}
    for row in csv_file:
        r = row.split(",")
        _labels[r[0].strip()] = int(r[1].strip())
    return _labels


def load_features(file_path):
    csv_file = open(file_path, "r")
    _features = {}
    for row in csv_file:
        r = row.split(",")
        parsed_data = []
        for d in r[1:]:
            parsed_data.append(float(d.strip()))
        _features[r[0]] = parsed_data
    return _features


"""
Load stuff from file
"""
features = load_features("data\\features.csv")
labels = load_classification("data\\classification.csv")
t_features = load_features("data\\test_features.csv")
t_labels = load_classification("data\\test_classification.csv")
training_features = []
training_labels = []
test_features = []
test_labels = []

"""
Parse/Transform
"""
for key in features.keys():
    feature = features[key]
    label = labels[key]
    training_features.append(feature)
    training_labels.append(transform_to_one_hot([label], hot_size=9)[0])

for key in t_features.keys():
    feature = t_features[key]
    label = t_labels[key]
    test_features.append(feature)
    test_labels.append(transform_to_one_hot([label], hot_size=9)[0])

"""
Create placeholders
"""
label_placeholder = tf.placeholder("float", [None, 9])
features_placeholder = tf.placeholder("float", [None, 8])

"""
Create and train dnn
"""
dnn = create_dnn([50, 50, 50], features_placeholder, n_features=8, n_classes=9, randomize_biases=False)
train_and_test_dnn(
    dnn,
    features_placeholder, label_placeholder,
    DataContainer(training_features, training_labels, 10),
    test_features, test_labels,
    epochs=250, learning_rate=0.0001
)
