import numpy as np

from helpers.labello_helpers import collect_image_data
from matplotlib import pylab as pl
from sklearn import tree
from random import shuffle

training_labello_features = np.array(collect_image_data("images\\training_labello\\*"))
training_other_features = np.array(collect_image_data("images\\training_other\\*"))
test_labello_features = np.array(collect_image_data("images\\test_labello\\*"))
test_other_features = np.array(collect_image_data("images\\test_other\\*"))

pl.figure(1)
pl.subplot(2, 2, 1)
pl.bar(range(2), [training_labello_features[:, 0].mean(), training_other_features[:, 0].mean()])
pl.subplot(2, 2, 2)
pl.bar(range(2), [training_labello_features[:, 1].mean(), training_other_features[:, 1].mean()], color='Green')
pl.subplot(2, 2, 3)
pl.bar(range(2), [training_labello_features[:, 2].mean(), training_other_features[:, 2].mean()], color='Blue')
pl.subplot(2, 2, 4)
pl.bar(range(2), [training_labello_features[:, 3].mean(), training_other_features[:, 3].mean()], color='Red')
#pl.figure(2)
#pl.scatter(training_labello_features[:, 0], training_labello_features[:, 3])
#pl.scatter(training_other_features[:, 0], training_other_features[:, 3])
pl.show()

"""
Labels: 0 (Labello), 1 (Not Labello)
"""

training_features = []
for i, features in enumerate(training_labello_features):
    training_features.append([0, features])

for i, features in enumerate(training_other_features):
    training_features.append([1, features])
shuffle(training_features)
features = []
labels = []
for f in training_features:
    labels.append(f[0])
    features.append(f[1])

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit((features), (labels))

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for test_features in test_labello_features:
    prediction = classifier.predict([test_features])
    print("Prediction: %d, actual: 0" % prediction)
    if prediction == 0:
        true_positive += 1
    else:
        false_negative += 1
        
for test_features in test_other_features:
    prediction = classifier.predict([test_features])
    print("Prediction: %d, actual: 1" % prediction)
    if prediction == 1:
        true_negative += 1
    else:
        false_positive += 1
        

print("TP %d, TN: %d, FP: %d, FN: %d" % (true_positive, true_negative, false_positive, false_negative))
