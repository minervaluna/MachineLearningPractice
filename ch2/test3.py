"""
示例：将带标签的样本数据绘制成散点图，选取第2和第3个特征值
结论：通过点的颜色区分类别，但还是很难得出结论性信息
"""

from ch2.kNN import *
import matplotlib.pyplot as plt

dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
plt.show()
