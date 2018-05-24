import tensorflow as tf
import random
import numpy as np


# TODO Use TF datasets! Also has CSV parser: https://www.tensorflow.org/get_started/datasets_quickstart
class DataContainer:
    batch_size: int
    batches: []
    current_batch: int

    def __init__(self, features, labels, batch_size=1) -> None:
        super().__init__()
        self.batches = []
        self.current_batch = 0
        data_length = len(features)
        if data_length != len(labels):
            raise ValueError("Features and labels have different lengths!")
        if batch_size < 1:
            raise ValueError("Batch size must be greater, than 0!")
        self.batch_size = batch_size
        for i in range(int(len(features) / batch_size)):
            last_index = (i + 1) * batch_size
            if last_index > data_length:
                last_index = data_length

            self.batches.append((features[i * batch_size:last_index], labels[i * batch_size:last_index]))

    def next(self):
        if self.current_batch >= len(self.batches):
            raise IndexError("No more batches!")
        batch = self.batches[self.current_batch]
        self.current_batch += 1
        return batch

    def has_next(self):
        if self.current_batch >= len(self.batches):
            return False
        return True

    def reset(self):
        self.current_batch = 0

    def get_once(self):
        feature_res = []
        class_res = []
        for i in self.batches:
            for j in i[0]:
                feature_res.append(j)
            for j in i[1]:
                class_res.append(j)
        return feature_res, class_res


def transform_to_one_hot(labels, **kwargs):
    """
    Create a list with the size of maximum value inside the parameter list.
    The one hot will be at the index of a label value.
    Example:

    For: labels = [1, 2, 4]
    Result will be: [[0,0,0,1],[0,0,1,0],[1,0,0,0]]

    The label values should be "compressed" (consecutive number only), this way there won't be elements which will
    never have the value 1.
    The labels parameter may also be a single integer, in this case hot_size is a must

    There's an optional parameter "hot_size", which sets the width of the result.
    Useful, if the labels array might not contain all the possible values.

    :param labels: array-like containing integers
    :return: matrix with the width of the max(labels) and length of len(labels)
    """
    if "hot_size" in kwargs:
        hot_size = kwargs["hot_size"]
    else:
        hot_size = max(labels) + 1

    if isinstance(labels, int):
        one_hot = [0] * hot_size
        one_hot[labels] = 1
        return one_hot

    result = []
    for label in labels:
        one_hot = [0] * hot_size
        one_hot[label] = 1
        result.append(one_hot)

    return result


def create_dnn(layer_neurons: [], features_placeholder, randomize_biases=False, **kwargs):
    # Check if the number of classes and features are provided explicitly
    # If so, add them to the list of neurons
    # This might be a stupid feature, but whatever.
    if "n_classes" in kwargs:
        layer_neurons.append(kwargs["n_classes"])
    if "n_features" in kwargs:
        layer_neurons.insert(0, kwargs["n_features"])

    current_data = features_placeholder
    nn_len = len(layer_neurons)
    for i in range(nn_len - 2):
        if randomize_biases:
            biases = tf.Variable(tf.random_normal([layer_neurons[i + 1]]))
        else:
            biases = tf.Variable(tf.ones([layer_neurons[i + 1]]))
        weights = tf.Variable(tf.random_normal([layer_neurons[i], layer_neurons[i + 1]]))
        # previous * weight + bias
        current_data = tf.add(
            tf.matmul(current_data, weights),
            biases
        )
        # Activation function using linear rectifier
        current_data = tf.nn.relu(current_data)

    # Compute the outputs without
    n_classes = layer_neurons[nn_len - 1]
    n_previous = layer_neurons[nn_len - 2]
    if randomize_biases:
        biases = tf.Variable(tf.random_normal([n_classes]))
    else:
        biases = tf.Variable(tf.ones([n_classes]))
    return tf.add(
        tf.matmul(
            current_data,
            tf.Variable(tf.random_normal([n_previous, n_classes]))
        ),
        biases
    )


def train_and_test_dnn(dnn, features_placeholder, label_placeholder, train_data: DataContainer,
                       test_data: DataContainer, epochs, learning_rate=0.001):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dnn, labels=label_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            while train_data.has_next():
                curr_x, curr_y = train_data.next()
                _, c = session.run(
                    [optimizer, cost],
                    feed_dict={features_placeholder: curr_x, label_placeholder: curr_y}
                )
                epoch_loss += c
            train_data.reset()
            print("Epoch %d/%d completed. Loss for this epoch was %.2f" % (epoch + 1, epochs, epoch_loss))

        # accuracy = session.run([dnn],
        #                        feed_dict={features_placeholder: test_features, label_placeholder: test_labels})
        # for r in accuracy:
        #     for i, r1 in enumerate(r):
        #         print(str(np.argmax(test_labels[i])) + " - " + str(np.argmax(r1)))
        correct = tf.equal(tf.argmax(dnn, 1), tf.argmax(label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        all_test_data = test_data.get_once()
        accuracy = accuracy.eval({features_placeholder: all_test_data[0], label_placeholder: all_test_data[1]})
        print("Accuracy: %.2f" % accuracy)
        return accuracy


class SampleSet:
    samples: []

    def __init__(self, labels, features, one_hot=True):
        self.samples = []
        if one_hot:
            hot_size = max(labels.values()) + 1
        else:
            hot_size = 0

        for k, v in labels.items():
            if one_hot:
                self.samples.append((k, transform_to_one_hot(v, hot_size=hot_size), features[k]))
            else:
                self.samples.append((k, v, features[k]))

    def get_data_containers(self, train_percentage=0.8, randomize=True, batch_size=10):
        if train_percentage >= 1 or train_percentage <= 0:
            raise ValueError("train_percentage has to be between 0 and 1")
        if randomize:
            randomized = random.sample(self.samples, len(self.samples))
        else:
            randomized = self.samples
        randomized = np.array(randomized, dtype=tuple)
        count = int(len(self.samples) * train_percentage)
        return (
            DataContainer(list(randomized[:count, 2]), list(randomized[:count, 1]), batch_size),
            DataContainer(list(randomized[count:, 2]), list(randomized[count:, 1]), batch_size)
        )


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
