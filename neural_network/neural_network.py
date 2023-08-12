from .layers import Layer
from .optimizers import Optimizer
from .losses import Loss

from .matrix import Matrix

class NeuralNetwork:
    ''' Neural Network. Deep Learning base model. '''
    
    def __init__(self, optimizer: Optimizer, loss: Loss):
        self.optimizer = optimizer
        
        self.loss_function = loss
        
        self.layers: list[Layer] = []
        
    def add(self, layer: Layer) -> None:
        ''' Add a layer to the neural network '''
        
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())
        
        layer.initialize(self.optimizer)
        
        self.layers.append(layer)
        
    def forward_pass(self, X: list[float]) -> Matrix:
        ''' Calculate the output of the NN '''
        
        output: list[Matrix] | Matrix = [Matrix(28, 28).fill(X)]
        
        for layer in self.layers:
            output = layer.forward(output)
        
        if isinstance(output, list):
            output = Matrix.join(output)
        
        return output
        
    def backward_pass(self, loss: Matrix) -> None:
        ''' Propagate the error backward through the NN '''
    
        for layer in reversed(self.layers):
            loss = layer.backward(loss)
        
    def fit(
        self, 
        X: list[list[float]], 
        y: list[list[float]], 
        epochs: int = 200, 
        batch_size: int = 200,
        max_no_change: int = 10, 
        tol: float = 1e-4
    ) -> None:
        ''' Train the neural network '''
        
        best_loss: float = float('inf')
        no_change: int = 0
        
        for curr_iter in range(epochs):
            loss_mean = 0.0
            
            for i in range(0, len(X), batch_size):
                batch = list(zip(X[i:i + batch_size], y[i:i + batch_size]))
                
                for Xi, yi in batch:
                    predicted = self.forward_pass(Xi)
                    
                    loss_mean += self.loss_function.loss(Matrix.from_list(yi), predicted)
                    loss_grad = self.loss_function.gradient(Matrix.from_list(yi), predicted)
                    
                    self.backward_pass(loss_grad)
                    
            loss_mean /= len(X)
            
            print(f'Iteration: {curr_iter + 1}, loss: {loss_mean}')
            
            if loss_mean > best_loss - tol:
                no_change += 1
            else:
                no_change = 0
                
            if loss_mean < best_loss:
                best_loss = loss_mean
            
            if no_change >= max_no_change:
                print(f'No change in loss for {max_no_change} iterations, stopping')
                    
                break