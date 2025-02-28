"""
示例：将不带标签的样本数据绘制成散点图
结论：难以辨识图中的点属于哪个样本分类
"""
from chapter2.kNN import *
import matplotlib.pyplot as plt

dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
plt.show()
