"""
归一化
"""
from chapter2.kNN import *
import matplotlib.pyplot as plt

dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(norm_mat[:, 0], norm_mat[:, 1], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
plt.show()