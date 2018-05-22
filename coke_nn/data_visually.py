from matplotlib import pylab as pl


def load_classification(file_path):
    csv_file = open(file_path, "r")
    _labels = {}
    for row in csv_file:
        r = row.split(",")
        _labels[r[0].strip()] = int(r[1].strip())
    return _labels


counts = [0] * 13
classification = load_classification("data\\classification.csv")
for k, v in classification.items():
    counts[v] += 1
print(counts)
pl.figure()
pl.bar(
    [
        "OK",
        "No bottle",
        "Cap",
        "Label",
        "Low",
        "High",
        "Cap, label",
        "Cap, low",
        "Cap, high",
        "Label, low",
        "Label, high",
        "Cap, label, low",
        "Cap, label, high",
    ],
    counts
)

pl.show()
