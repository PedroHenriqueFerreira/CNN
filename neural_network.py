from typing import Optional, Any

from matrix import Matrix
from layers import Layer
from optimizers import Optimizer
from losses import Loss

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        loss: Loss,
    ):
        self.layers: list[Layer] = []
        
        self.optimizer = optimizer
        self.loss = loss
    
    def add(self, layer: Layer) -> None:
        
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].output_shape)
            
        layer.initialize(self.optimizer)
        
        self.layers.append(layer)
        
    def forward(self, input_value: Matrix, training: bool = True) -> Matrix:
        for layer in self.layers:
            input_value = layer.forward(input_value, training)
            
        return input_value
    
    def backward(self, output_gradient: Matrix, training: bool = True) -> Matrix:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, training)
            
        return output_gradient
    
    def train_on_batch(self, X: Matrix, y: Matrix) -> float:
        y_pred = self.forward(X)
        self.backward(self.loss.gradient(y, y_pred))
        
        return self.loss(y, y_pred).mean() # type: ignore
    
    def not_train_on_batch(self, X: Matrix, y: Matrix) -> tuple[float, Matrix]:
        y_pred = self.forward(X, training=False)
        input_gradient = self.backward(self.loss.gradient(y, y_pred), training=False)
        
        return self.loss(y, y_pred).mean(), input_gradient

    def fit(
        self, 
        X: Matrix, 
        y: Matrix, 
        epochs: int, 
        batch_size: int, 
        X_val: Optional[Matrix] = None,
        y_val: Optional[Matrix] = None
    ) -> None:
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            loss: list[float] = []
            
            for X_batch, y_batch in zip(X.split(batch_size), y.split(batch_size)):
                loss.append(self.train_on_batch(X_batch, y_batch))
                
            print(f'Loss: {sum(loss) / len(loss)}')
            
            if X_val is not None and y_val is not None:
                print(f'Validation Loss: {self.not_train_on_batch(X_val, y_val)}')
    
    def summary(self, col_width: int = 25) -> None:
        layer_name: Any = 'Layer Name'.ljust(col_width)
        output_shape: Any = 'Output Shape'.ljust(col_width)
        params: Any = 'Params'
        
        print(f'{layer_name} {output_shape} {params}')
        print('-' * (col_width * 3 + 2))
        
        layer_name = 'InputLayer'.ljust(col_width)
        output_shape = str((None, *self.layers[0].input_shape)).ljust(col_width)
        params = 0
        
        print(f'{layer_name} {output_shape} {params}')
        
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            output_shape = (None, *layer.output_shape)
            params = layer.parameters()
            
            print(f'{layer_name.ljust(col_width)} {str(output_shape).ljust(col_width)} {params}')
            
        print()
        print(f'Total params: {sum(layer.parameters() for layer in self.layers)}')