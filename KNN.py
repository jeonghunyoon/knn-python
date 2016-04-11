import numpy

def classify(x, datasets, labels, k):
    datasets_size = datasets.shape[0]
    x = numpy.tile(x, (datasets_size, 1))
