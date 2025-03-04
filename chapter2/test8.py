from chapter2.kNN import handwriting_class_test

handwriting_class_test()

"""
思考：
1.影响错误率的因素；
2.k值对结果的影响；
3.样本的改变对结果的影响；

优化：
knn算法效率不高，将测试数据load进测试向量时，需要比较大的内存空间，
为此引入下一章的k决策树，就是k-邻近算法的优化版本，可以大量节省计算开销
"""