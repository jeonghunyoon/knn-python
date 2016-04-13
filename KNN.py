from numpy import *
import operator

"""
(sample)
x = [0, 0.5]
datasets = array([[1,1],[1,0],[0,1],[0,0]])
labels = array(['A', 'A', 'B', 'B'])
k = 2
"""
def classify(x, datasets, labels, k):
    datasets_size = datasets.shape[0]
    sum_sqr_diff_matrix = ((tile(x, (datasets_size, 1)) - datasets) ** 2).sum(axis=1)
    dist_matrix = sum_sqr_diff_matrix ** 0.5
    sorted_distance_idx = dist_matrix.argsort(kind='quicksort')

    label_count = dict()
    for i in range(k):
        selected_label = labels[sorted_distance_idx[i]]
        label_count[selected_label] = label_count.get(selected_label, 0) + 1

    class_predict = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    return class_predict[0][0]
