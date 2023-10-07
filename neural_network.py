from typing import Optional

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
        validation: Optional[tuple[Matrix, Matrix]] = None
    ):
        self.layers: list[Layer] = []
        
        self.optimizer = optimizer
        self.loss = loss
        self.validation = validation
    
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
            output_gradient = layer.backward(output_gradient)
            
        return output_gradient
    
    def fit(self, X: Matrix, y: Matrix, n_epochs: int, batch_size: int) -> None:
        ...
    
    def summary(self, col_width: int = 25) -> None:
        layer_name: str = 'Layer Name'.ljust(col_width)
        output_shape: str | tuple[int | None, ...] = 'Output Shape'.ljust(col_width)
        params: str | int = 'Params'
        
        print(f'{layer_name} {output_shape} {params}')
        print()
        
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            output_shape = (None, *layer.output_shape)
            params = layer.parameters()
            
            layer_name = layer_name.ljust(col_width)
            output_shape = str(output_shape).ljust(col_width)
            params = str(params)
            
            print(f'{layer_name.ljust(col_width)} {str(output_shape).ljust(col_width)} {params}')
            
        print()
        print(f'Total params: {sum(layer.parameters() for layer in self.layers)}')