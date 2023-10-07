from layers import Activation
from matrix import Matrix

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
        return x.exp() / x.exp().sum()
    
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