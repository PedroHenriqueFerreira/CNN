from matrix import Matrix

class Optimizer:
    ''' Base class for all optimizers '''

    def __init__(self):
        ''' Initialize the optimizer '''
        
        raise NotImplementedError()

    def copy(self) -> 'Optimizer':
        ''' Return a copy of the optimizer '''
        
        raise NotImplementedError()

    def update(self, param: Matrix, gradient: Matrix) -> Matrix:
        ''' Update the parameter of the neural network '''
        
        raise NotImplementedError()

class SGDOptimizer(Optimizer):
    ''' Stochastic Gradient Descent optimizer '''
    
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.param_update: Matrix | None = None
    
    def copy(self) -> 'SGDOptimizer':    
        return SGDOptimizer(self.learning_rate, self.momentum)
    
    def update(self, param: Matrix, gradient: Matrix) -> Matrix:
        if self.param_update is None:
            self.param_update = param.zeros()
            
        self.param_update = self.momentum * self.param_update + (1 - self.momentum) * gradient
        
        return param - self.learning_rate * self.param_update
    
class AdamOptimizer(Optimizer):
    ''' Adaptive Moment Estimation optimizer '''
    
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
      
        self.m: Matrix | None = None
        self.v: Matrix | None = None

    def copy(self) -> 'AdamOptimizer':
        return AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon)

    def update(self, param: Matrix, gradient: Matrix) -> Matrix:
        if self.m is None or self.v is None:
            self.m = param.zeros()
            self.v = param.zeros()
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        
        return param - self.learning_rate * m_hat / (self.epsilon + (v_hat ** 0.5))
