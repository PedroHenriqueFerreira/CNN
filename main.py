from neural_network import NeuralNetwork
from optimizers import AdamOptimizer
from losses import CrossEntropyLoss, MeanSquaredLoss
from activations import LeakyReLU, TanH, Sigmoid
from layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten

from time import time

init = time()


def generator():
    g_optimizer = AdamOptimizer(learning_rate=0.0005)
    g_loss = MeanSquaredLoss()
    
    g = NeuralNetwork(g_optimizer, g_loss)

    g.add(Dense(128*8*8, input_shape=(100,)))
    g.add(LeakyReLU())
    g.add(Reshape((128, 8, 8)))
    
    g.add(UpSampling2D())
    g.add(Conv2D(64, (3, 3), padding='same'))
    g.add(LeakyReLU())
    g.add(BatchNormalization())
    
    g.add(UpSampling2D())
    g.add(Conv2D(32, (3, 3), padding='same'))
    g.add(LeakyReLU())
    g.add(BatchNormalization())
    
    g.add(UpSampling2D())
    g.add(Conv2D(16, (3, 3), padding='same'))
    g.add(LeakyReLU())
    g.add(BatchNormalization())
    
    g.add(Conv2D(3, (3, 3), padding='same'))
    
    g.add(TanH())

    g.summary()
    
    return g
    
def descriminator():
    d_optimizer = AdamOptimizer(learning_rate=0.0005)
    d_loss = CrossEntropyLoss()
    
    d = NeuralNetwork(d_optimizer, d_loss)
    
    d.add(Conv2D(16, (3, 3), padding='same', input_shape=(3, 64, 64)))
    d.add(LeakyReLU())
    d.add(MaxPooling2D((2, 2), 2))
    d.add(BatchNormalization())
    d.add(Dropout(0.25))
    
    d.add(Conv2D(32, (3, 3), padding='same'))
    d.add(LeakyReLU())
    d.add(MaxPooling2D((2, 2), 2))
    d.add(BatchNormalization())
    d.add(Dropout(0.25))
    
    d.add(Conv2D(64, (3, 3), padding='same'))
    d.add(LeakyReLU())
    d.add(MaxPooling2D((2, 2), 2))
    d.add(BatchNormalization())
    d.add(Dropout(0.25))
    
    d.add(Flatten())
    d.add(Dense(256))
    d.add(LeakyReLU())
    d.add(Dropout(0.25))
    d.add(Dense(1))
    d.add(Sigmoid())
    
    d.summary()
    
    return d

d = descriminator()
g = generator()