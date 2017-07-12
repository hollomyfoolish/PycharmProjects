import numpy as np

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T
# syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1
syn0 = np.zeros((3, 4))
# syn1 = np.zeros((4, 1))
print syn0
print '======================='
print syn1
print '======================='
for j in xrange(60000):
    l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))

    l2_delta = (y - l2) * (l2 * (1 - l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))

    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
print X
print l1
print l2
print y