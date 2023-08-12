from neural_network import NeuralNetwork

from neural_network.layers import Conv2D, MaxPooling2D, Flatten, Dense
from neural_network.activations import ReLU, Softmax
from neural_network.losses import CrossEntropyLoss
from neural_network.optimizers import AdamOptimizer

from database import Database
from database.encoders import OneHotEncoder

db = Database.from_csv('mnist_train.csv')

encoder = OneHotEncoder()

db[0] = encoder.fit_transform(db[0])

X_train = db[10:]
y_train = db[0:10]

nn = NeuralNetwork(AdamOptimizer(), CrossEntropyLoss())

nn.add(Conv2D(8, (3, 3), input_shape=(1, 28, 28)))
nn.add(ReLU())
nn.add(MaxPooling2D((2, 2), 2))
nn.add(Conv2D(32, (3, 3)))
nn.add(ReLU())
nn.add(MaxPooling2D((2, 2), 2))
nn.add(Flatten())
nn.add(Dense(20))
nn.add(ReLU())
nn.add(Dense(10))
nn.add(Softmax())

print('Training...')
nn.fit(X_train.values, y_train.values)