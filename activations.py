from math import exp

from .matrix import Matrix

class Activation:
    ''' Base class for all activations '''
    
    def __call__(self, x: Matrix) -> Matrix:
        ''' Return the activation of the matrix '''
        
        raise NotImplementedError()
    
    def gradient(self, x: Matrix) -> Matrix:
        ''' Return the gradient of the matrix '''
        
        raise NotImplementedError()

class Sigmoid(Activation):
    ''' Sigmoid activation class '''
    
    def __call__(self, x: Matrix) -> Matrix:
        return 1 / (1 + (-x).exp())
    
    def gradient(self, x: Matrix) -> Matrix:
        return self(x) * (1 - self(x))

class TanH(Activation):
    ''' TanH activation class '''
      
    def __call__(self, x: Matrix) -> Matrix:
        return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
    
    def gradient(self, x: Matrix) -> Matrix:
        return 1 - self(x) ** 2

class ReLU(Activation):
    ''' ReLU activation class '''
    
    def __call__(self, x: Matrix) -> Matrix:
        return x.map(lambda i: max(0, i))
    
    def gradient(self, x: Matrix) -> Matrix:
        return x.map(lambda i: int(i > 0))

class Softmax(Activation):
    ''' Softmax activation class '''
    
    def __call__(self, x: Matrix) -> Matrix:
        return exp(x) / x.exp().sum()
    
    def gradient(self, x: Matrix) -> Matrix:
        return self(x) * (1 - self(x))
    
class LeakyReLU(Activation):
    ''' LeakyReLU activation class '''
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x: Matrix) -> Matrix:
        return x.map(lambda i: max(self.alpha * i, i))
    
    def gradient(self, x: Matrix) -> Matrix:
        return x.map(lambda i: self.alpha if i < 0 else 1)