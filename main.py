from activations import Sigmoid, TanH, Identity, Softmax
from matrix import Matrix

sigmoid = Sigmoid()

m = Matrix.load([[1, 2], [3, 4]])

print(sigmoid(m))