from helpers.nn_helper import *
import tensorflow as tf


def load_classification(file_path):
    csv_file = open(file_path, "r")
    _labels = {}
    for row in csv_file:
        r = row.split(",")
        _labels[r[0].strip()] = float(r[1].strip())
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


features = load_features("data.csv")
labels = load_classification("simple_classification.csv")
training_features = []
training_labels = []

for key, label in features:
    feature = features.get(key)
    if feature is None:
        print("Didn't find features of %s" % key)
        continue
    training_features.append(feature)
    training_labels.append(transform_to_one_hot([label], hot_size=9))

epochs = 10

x = tf.placeholder("float", [None, 8])
label_placeholder = tf.placeholder("float")

dnn = create_dnn([10, 10, 10], n_features=8, n_classes=9)
train_dnn(dnn, label_placeholder, DataContainer(training_features, training_labels, 10), 20)
