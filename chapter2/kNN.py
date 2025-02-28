import operator

from numpy import *


# 已知数据集和标签
def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# knn分类算法
def classify0(x, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = tile(x, (dataset_size, 1)) - dataset
    square_diff = diff_mat ** 2
    square_distances = square_diff.sum(axis=1)
    distances = square_distances ** 0.5
    sorted_indices = square_distances.argsort()
    class_counts = {}
    for i in range(k):
        vote_label = labels[sorted_indices[i]]
        class_counts[vote_label] = class_counts.get(vote_label, 0) + 1
    sorted_class_counts = sorted(class_counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_counts[0][0]

def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))

    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector

def auto_norm(dataset):
    min_vals = dataset.min(axis=0)  # 获取每列的最小值
    max_vals = dataset.max(axis=0)  # 获取每列的最大值
    ranges = max_vals - min_vals  # 计算每列的范围（极差）
    norm_dataset = (dataset - min_vals) / ranges  # 归一化
    return norm_dataset, ranges, min_vals

def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vectors = int(m * ho_ratio)
    error_count = 0
    for i in range(num_test_vectors):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vectors:m, :], dating_labels[num_test_vectors:m], 3)
        print('the result of the classify is:', classifier_result, 'the real one is:', dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1
    print('the total error rate is:', error_count / float(num_test_vectors))

def classify_person():
    results = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('You will probably like this person:', results[classifier_result - 1])