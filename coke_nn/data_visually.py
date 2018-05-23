from matplotlib import pylab as pl


def load_classification(file_path):
    csv_file = open(file_path, "r")
    _labels = {}
    for row in csv_file:
        r = row.split(",")
        _labels[r[0].strip()] = int(r[1].strip())
    return _labels


counts = [0] * 5
classification = load_classification("data\\classification_cap_label.csv")
for k, v in classification.items():
    counts[v] += 1
print(counts)
pl.figure()
pl.subplot(2, 1, 1)
pl.bar(
    [
        "OK",
        "No bottle",
        "Cap",
        "Label",
        "Cap, label"
    ],
    counts
)
counts = [0] * 3
classification = load_classification("data\\classification_fluid.csv")
for k, v in classification.items():
    counts[v] += 1
print(counts)
pl.subplot(2, 1, 2)
pl.bar(
    [
        "OK",
        "Low",
        "High",
    ],
    counts
)
pl.show()
