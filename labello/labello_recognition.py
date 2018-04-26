import numpy as np

from helpers.labello_helpers import collect_image_data
from matplotlib import pylab as pl

training_labello_features = np.array(collect_image_data("images\\training_labello\\*"))
training_other_features = np.array(collect_image_data("images\\training_other\\*"))

pl.figure(1)
pl.subplot(2, 2, 1)
pl.bar(range(2), [training_labello_features[:, 0].mean(), training_other_features[:, 0].mean()])
pl.subplot(2, 2, 2)
pl.bar(range(2), [training_labello_features[:, 1].mean(), training_other_features[:, 1].mean()], color='Green')
pl.subplot(2, 2, 3)
pl.bar(range(2), [training_labello_features[:, 2].mean(), training_other_features[:, 2].mean()], color='Blue')
pl.subplot(2, 2, 4)
pl.bar(range(2), [training_labello_features[:, 3].mean(), training_other_features[:, 3].mean()], color='Red')
# pl.figure(2)
# pl.scatter(training_labello_features[:, 0], training_labello_features[:, 3])
# pl.scatter(training_other_features[:, 0], training_other_features[:, 3])
pl.show()
