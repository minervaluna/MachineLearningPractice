from numpy import *

# 随机生成一个4x4的二位数组
array = random.rand(4, 4)
print('array:')
print(array)

# 将二位数组转成矩阵
matrix = matrix(array)
print('matrix:')
print(matrix)

# 逆矩阵
invMatrix = matrix.I
print('invMatrix:')
print(invMatrix)

# 原矩阵 * 逆矩阵，发现对角线为1，其余为0，计算机输出不为0是因为精度问题
multiply = matrix * invMatrix
print('multiply:')
print(multiply)
