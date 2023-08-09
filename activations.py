from math import exp

from layers import Activation
from matrix import Matrix

class Identity(Activation):
    ''' Identity activation class '''
    
    def activation(self, x: Matrix) -> Matrix:
        return x.map(lambda i: i)

    def gradient(self, x: Matrix) -> Matrix:
        return x.map(lambda i: 1)

class Sigmoid(Activation):
    ''' Sigmoid activation class '''
    
    def sigmoid(self, x: float) -> float:
        ''' Sigmoid function with overflow protection '''
        
        try:
            return 1 / (1 + exp(-x))
        
        except OverflowError:
            return 0
    
    def activation(self, x: Matrix) -> Matrix:
        return x.map(self.sigmoid)
    
    def gradient(self, x: Matrix) -> Matrix:
        return self.activation(x) * (1 - self.activation(x))

class TanH(Activation):
    ''' TanH activation class '''
        
    def tanh(self, x: float) -> float:
        ''' Tanh function with overflow protection '''
        
        try:
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        except OverflowError:
            return 1 if x > 0 else -1
        
    def activation(self, x: Matrix) -> Matrix:
        return x.map(self.tanh)
    
    def gradient(self, x: Matrix) -> Matrix:
        return 1 - self.activation(x) ** 2

class ReLU(Activation):
    ''' ReLU activation class '''
    
    def activation(self, x: Matrix) -> Matrix:
        return x.map(lambda i: max(0, i))
    
    def gradient(self, matrix: Matrix) -> Matrix:
        return matrix.map(lambda i: int(i > 0))

class Softmax(Activation):
    ''' Softmax activation class '''

    def softmax(self, x: float) -> float:
        ''' Softmax function with overflow protection '''
        
        try:
            exp_sum = self.input.map(lambda x: exp(x)).sum()
        except OverflowError:
            exp_sum = float('inf')
        
        try:
            return exp(x) / (exp_sum or 1)
        except OverflowError:
            return 1
    
    def activation(self, x: Matrix) -> Matrix:
        return x.map(self.softmax)
    
    def gradient(self, x: Matrix) -> Matrix:
        return self.activation(x) * (1 - self.activation(x))
