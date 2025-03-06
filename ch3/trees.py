from math import log

"""
计算给定数据集的香农熵
"""
def calc_shannon_entropy(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        curr_label = feat_vec[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy

def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels