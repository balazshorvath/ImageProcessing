import tensorflow as tf
import nu


class DataContainer:
    batch_size: int
    batches = []
    current_batch: 0

    def __init__(self, features, labels, batch_size=1) -> None:
        super().__init__()
        data_length = len(features)
        if data_length != len(labels):
            raise ValueError("Features and labels have different lengths!")
        if batch_size < 1:
            raise ValueError("Batch size must be greater, than 0!")
        self.batch_size = batch_size
        for i in range(len(features / batch_size)):
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


def transform_to_one_hot(labels: [], **kwargs):
    """
    Create a list with the size of maximum value inside the parameter list.
    The one hot will be at the index of a label value.
    Example:

    For: labels = [1, 2, 4]
    Result will be: [[0,0,0,1],[0,0,1,0],[1,0,0,0]]

    The label values should be "compressed" (consecutive number only), this way there won't be elements which will
    never have the value 1.

    There's an optional parameter "hot_size", which sets the width of the result.
    Useful, if the labels array might not contain all the possible values.

    :param labels: array-like containing integers
    :return: matrix with the width of the max(labels) and length of len(labels)
    """
    if "hot_size" is kwargs:
        hot_size = kwargs["hot_size"]
    else:
        hot_size = max(labels)
    result = []
    for label in labels:
        one_hot = [0] * hot_size
        one_hot[label] = 1
        result.append(one_hot)

    return result


def create_dnn(layer_neurons: [], randomize_biases=False, **kwargs):
    # Check if the number of classes and features are provided explicitly
    # If so, add them to the list of neurons
    # This might be a stupid feature, but whatever.
    if "n_classes" in kwargs:
        layer_neurons.append(kwargs["n_classes"])
    if "n_features" in kwargs:
        layer_neurons.insert(0, kwargs["n_features"])

    current_data = tf.placeholder("float", [None, len(layer_neurons[0])])
    nn_len = len(layer_neurons)
    for i in range(nn_len - 2):
        if randomize_biases:
            biases = tf.Variable(tf.random_normal([layer_neurons[i]]))
        else:
            biases = tf.Variable(tf.ones([layer_neurons[i]]))
        # previous * weight + bias
        current_data = tf.add(
            tf.matmul(
                current_data,
                tf.Variable(tf.random_normal([layer_neurons[i], layer_neurons[i + 1]]))
            ),
            biases
        )
        # Activation function using linear rectifier
        current_data = tf.nn.relu(current_data)

    # Compute the outputs without
    n_classes = layer_neurons[nn_len - 1]
    n_previous = layer_neurons[nn_len - 2]
    if randomize_biases:
        biases = tf.Variable(tf.random_normal(n_previous))
    else:
        biases = tf.Variable(tf.ones(n_previous))
    return tf.add(
        tf.matmul(
            current_data,
            tf.Variable(tf.random_normal([n_previous, n_classes]))
        ),
        biases
    )


def train_dnn(dnn, label_placeholder, train_data: DataContainer, epochs, learning_rate=0.001):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(dnn, label_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            while train_data.has_next():
                x, y = train_data.next()
                _, c = session.run([optimizer, cost], feed_dict={x: x, y: y})
                epoch_loss = c
            print("Epoch %d/%d completed. Loss for this epoch was %.2f" % (epoch, epochs, epoch_loss))
