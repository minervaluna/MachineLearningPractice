"""
分类越多 => 熵越大 => 数据越混乱
"""
from ch3 import trees

dataset, labels = trees.create_dataset()
entropy = trees.calc_shannon_entropy(dataset)
print(entropy)

dataset[0][-1] = 'maybe'
entropy = trees.calc_shannon_entropy(dataset)
print(entropy)