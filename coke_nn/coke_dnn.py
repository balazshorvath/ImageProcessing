from helpers.nn_helper import *
import tensorflow as tf
import time

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
cap_label_training, cap_label_test = cap_and_label_set.get_data_containers(0.8, randomize=True)

print("Data ready.")
"""
Create placeholders
"""
fluid_classes_count = 3
fluid_feature_count = 1
cap_label_classes_count = 5
cap_label_feature_count = 6
fluid_label_placeholder = tf.placeholder("float", [None, fluid_classes_count])
fluid_features_placeholder = tf.placeholder("float", [None, fluid_feature_count])
cap_label_label_placeholder = tf.placeholder("float", [None, cap_label_classes_count])
cap_label_features_placeholder = tf.placeholder("float", [None, cap_label_feature_count])

accuracies = []

for i in range(20):
    """
    Create and train fluid dnn
    """
    print("Initializing DNN.")
    dnn = create_dnn([50, 50, 50], fluid_features_placeholder, n_features=fluid_feature_count,
                     n_classes=fluid_classes_count, randomize_biases=False)
    print("%d Initializing fluid DNN finished." % i)
    print("%d Training and testing fluid DNN." % i)
    accuracy = train_and_test_dnn(
        dnn,
        fluid_features_placeholder, fluid_label_placeholder,
        fluid_training, fluid_test,
        epochs=250, learning_rate=0.0001
    )
    accuracies.append(accuracy)
    print("%d Training fluid DNN finished." % i)

for i in range(20):
    """
    Create and train cap and label dnn
    """
    print("Initializing DNN.")
    dnn = create_dnn([50, 50, 50], cap_label_features_placeholder, n_features=cap_label_feature_count,
                     n_classes=cap_label_classes_count, randomize_biases=False)
    print("Initializing cap and label DNN finished.")
    print("Training and testing cap and label DNN.")
    accuracy = train_and_test_dnn(
        dnn,
        cap_label_features_placeholder, cap_label_label_placeholder,
        cap_label_training, cap_label_test,
        epochs=500, learning_rate=0.0001
    )
    accuracies.append(accuracy)
    print("%d Training cap and label DNN finished." % i)
print(accuracies)
