"""
通过mock数据集来演示knn算法
"""
from ch2 import kNN

group, labels = kNN.create_dataset()

label = kNN.classify0([0.2, 0.4], group, labels, 3)
print(label)
