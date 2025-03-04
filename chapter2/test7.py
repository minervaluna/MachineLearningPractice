from chapter2.kNN import img_to_vector

test_vector = img_to_vector('testDigits/0_13.txt')
result = test_vector[0, 0:31]
print(result)
result = test_vector[0, 32:63]
print(result)