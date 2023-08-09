from matrix import Matrix

from optimizers import Optimizer

class Layer:
    ''' Abstract class for layers '''
    
    def initialize(self, optimizer: Optimizer) -> None:
        ''' Initialize the layer '''
    
    def forward(self, input): 
        ''' Propagates input forward '''
        
        raise NotImplementedError()
    
    def backward(self, output_gradient):
        ''' Propagates gradient backward '''
        
        raise NotImplementedError()
    
class Dense(Layer):
    ''' Fully connected layer '''
    
    def __init__(self, input_size: int, output_size: int):
        ''' Initialize a fully connected layer '''
        
        self.input_size = input_size
        self.output_size = output_size
        
    def initialize(self, optimizer: Optimizer) -> None:
        limit = 1 / self.input_size ** 0.5
        
        self.weights = Matrix(self.output_size, self.input_size).randomize(-limit, limit)
        self.bias = Matrix(self.output_size, 1).zeros()
        
        self.weights_optimizer = optimizer.copy()
        self.bias_optimizer = optimizer.copy()
        
    def forward(self, input: Matrix) -> Matrix:
        self.input = input
        
        return self.weights @ input + self.bias
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient
        
        self.weights = self.weights_optimizer.update(self.weights, weights_gradient)
        self.bias = self.bias_optimizer.update(self.bias, output_gradient)
        
        return input_gradient

class Activation(Layer):
    ''' Activation layer '''
    
    def activation(self, x: Matrix) -> Matrix:
        ''' Return the activation of x '''
        
        raise NotImplementedError()
    
    def gradient(self, x: Matrix) -> Matrix:
        ''' Return the gradient of x '''
        
        raise NotImplementedError()
    
    def forward(self, input: Matrix) -> Matrix:
        self.input = input
        
        return self.activation(input)
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        return self.gradient(self.input) * output_gradient
    