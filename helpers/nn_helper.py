import tensorflow as tf


def create_dnn(data, layer_neurons: [], randomize_biases=False, **kwargs):
    # Check if the number of classes and features are provided explicitly
    # If so, add them to the list of neurons
    # This might be a stupid feature, but whatever.
    if "n_classes" in kwargs:
        layer_neurons.append(kwargs["n_classes"])
    if "n_features" in kwargs:
        layer_neurons.insert(0, kwargs["n_features"])

    current_data = data
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
