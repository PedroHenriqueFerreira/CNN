from neural_network import NeuralNetwork
from optimizers import AdamOptimizer
from losses import CrossEntropyLoss, MeanSquaredLoss
from activations import LeakyReLU, TanH, Sigmoid
from layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
    
from matrix import Matrix
    
def discriminator():
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
    
    return d

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
    
    return g

def read_images(src: str) -> Matrix:
    from os import listdir
    
    images = []
    
    for item in listdir(src):
        images.append(Matrix.from_image(src + '/' + item).data)
        
    matrix = Matrix.load(images)

    return (matrix / 255) * 2 - 1

images = read_images('./images')

d = discriminator()
g = generator()

epochs = 10
batch_size = 2

def run() -> None:
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        
        for i, batch in enumerate(images.split(batch_size)):
            print(f'Batch {i + 1}/{images.shape[0] // batch_size}')
            
            noise = Matrix(batch_size, 100).randomize(-1, 1)
            
            print('Generating Images...')
            generated_images = g.forward(noise, False)
            print('Images Generated!')
            
            y_fake = Matrix(batch_size, 1).randomize(0, 0.05)
            y_real = Matrix(batch_size, 1).randomize(0.95, 1)
            
            print('Training Discriminator...')
            d.train_on_batch(batch, y_real)
            d.train_on_batch(generated_images, y_fake)
            print('Discriminator Trained!')
            
            noise = Matrix(batch_size, 100).randomize(-1, 1)
            
            print('Training Generator...')
            _, err = d.not_train_on_batch(g.forward(noise), y_real)
            g.backward(err)
            print('Generator Trained!')
        
            print('Generating Image...')
    
            output = ((g.forward(noise, False) + 1) / 2) * 255
            
            output[0].to_image(f'./generated/epoch_{epoch}_batch_{i}.png')
            
            print('Image Generated!')
        
run()