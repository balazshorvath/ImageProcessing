from helpers.nn_helper import *
import tensorflow as tf

print("Loading and preparing data.")
features = load_features("data/features.csv")
fluid_features = {}
cap_label_features = {}

for k, v in features.items():
    fluid_features[k] = [v[len(v) - 1]]
    cap_label_features[k] = v[:len(v) - 1]

fluid_set = SampleSet(load_classification("data/classification_fluid.csv"), fluid_features)
cap_and_label_set = SampleSet(load_classification("data/classification_cap_label.csv"), cap_label_features)

fluid_training, fluid_test = fluid_set.get_data_containers(0.8, randomize=True)
cap_and_label_training, cap_and_label_test = cap_and_label_set.get_data_containers(0.8, randomize=True)

print(cap_and_label_test.get_once())

print("Data ready.")

print("Initializing DNN.")
"""
Create placeholders
"""
label_placeholder = tf.placeholder("float", [None, 3])
features_placeholder = tf.placeholder("float", [None, 1])

"""
Create and train dnn
"""
dnn = create_dnn([50, 50, 50], features_placeholder, n_features=1, n_classes=3, randomize_biases=False)
print("Initializing DNN finished.")
print("Training and testing DNN.")
train_and_test_dnn(
    dnn,
    features_placeholder, label_placeholder,
    fluid_training,
    fluid_test,
    epochs=250, learning_rate=0.0001
)
