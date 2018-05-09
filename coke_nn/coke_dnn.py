from helpers.nn_helper import create_dnn
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


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

x = tf.placeholder("float", [None, 6])
y = tf.placeholder("float", [None, 10])
input_data.read_data_sets("tmp/mnist", one_hot=True).train.next_batch()
