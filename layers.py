from typing import Literal, Any

from matrix import Matrix
from optimizers import Optimizer

class Layer:
    ''' Abstract class for layers '''
    
    def set_input_shape(self, input_shape: tuple[int, ...] | None) -> None:
        ''' Sets the input shape of the layer '''
        
        self.input_shape = input_shape
    
    def output_shape(self) -> tuple[int, ...]:
        ''' Returns the shape of the output '''
        
        raise NotImplementedError()
    
    def initialize(self, optimizer: Optimizer) -> None:
        ''' Initialize the layer '''
    
        raise NotImplementedError()
    
    def forward(self, input: Any) -> Any: 
        ''' Propagates input forward '''
        
        raise NotImplementedError()
    
    def backward(self, output_gradient: Any) -> Any:
        ''' Propagates gradient backward '''
        
        raise NotImplementedError()
    
class Dense(Layer):
    ''' Fully connected layer '''
    
    def __init__(self, units: int, input_shape: tuple[int, ...] | None = None):
        ''' Initialize a fully connected layer '''
        
        self.units = units
        self.input_shape = input_shape
    
    def output_shape(self) -> tuple[int]:
        return (self.units, )
    
    def initialize(self, optimizer: Optimizer) -> None:
        assert self.input_shape is not None
        
        limit = 1 / self.input_shape[0] ** 0.5
        
        self.weights = Matrix(self.units, self.input_shape[0]).randomize(-limit, limit)
        self.bias = Matrix(self.units, 1).zeros()
        
        self.weights_optimizer = optimizer.copy()
        self.bias_optimizer = optimizer.copy()
        
    def forward(self, input_value: Matrix) -> Matrix:
        self.input_value = input_value
        
        return self.weights @ input_value + self.bias
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        weights_gradient = output_gradient @ self.input_value.T
        input_gradient = self.weights.T @ output_gradient
        
        self.weights = self.weights_optimizer.update(self.weights, weights_gradient)
        self.bias = self.bias_optimizer.update(self.bias, output_gradient)
        
        return input_gradient

class Conv2D(Layer):
    ''' Convolutional layer '''
    
    def __init__(
        self, 
        kernel_shape: tuple[int, int], 
        filters: int,
        input_shape: tuple[int, int, int] | None = None, 
        stride: int = 1,
        padding: Literal['valid', 'same'] = 'valid'
    ):
        self.kernel_shape = kernel_shape
        self.filters = filters
        
        self.input_shape = input_shape
        self.stride = stride
        self.padding = padding
    
    def get_padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        ''' Returns the padding size '''
        
        assert self.input_shape is not None
        
        _, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        match self.padding:
            case 'valid':
                v_pad = (0, 0)
                h_pad = (0, 0)
            case 'same':
                v_total_pad = (input_height - 1) * self.stride - input_height + kernel_height
                h_total_pad = (input_width - 1) * self.stride - input_width + kernel_width
                
                v_pad = (v_total_pad // 2, v_total_pad - v_total_pad // 2)
                h_pad = (h_total_pad // 2, h_total_pad - h_total_pad // 2)
    
        return (v_pad, h_pad)
    
    def output_shape(self) -> tuple[int, ...]:
        assert self.input_shape is not None
        
        _, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        v_pad, h_pad = self.get_padding()
        
        height = (input_height + sum(v_pad) - kernel_height) // self.stride + 1
        width = (input_width + sum(h_pad) - kernel_width) // self.stride + 1
        
        return (self.filters, height, width)
    
    def initialize(self, optimizer: Optimizer) -> None:
        assert self.input_shape is not None
        
        channels, _, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        limit = 1 / (kernel_height * kernel_width) ** 0.5
        
        self.kernels: list[list[Matrix]] = [[] for _ in range(self.filters)]
        
        for filter_kernels in self.kernels:
            for _ in range(channels):
                kernel = Matrix(kernel_height, kernel_width).randomize(-limit, limit)
                
                filter_kernels.append(kernel)
        
        self.biases: list[float] = [0] * self.filters
        
        self.kernels_optimizer = optimizer.copy()
        self.bias_optimizer = optimizer.copy()
    
    def forward(self, input_value: list[Matrix]) -> list[Matrix]:
        assert self.input_shape is not None
        
        v_pad, h_pad = self.get_padding()
        self.input_value = [channel.expand(v_pad, h_pad) for channel in input_value]
        
        _, output_height, output_width = self.output_shape()
        self.output_value = [Matrix(output_height, output_width) for _ in range(self.filters)]
        
        channels, _, _ = self.input_shape
        
        for i in range(self.filters):
            for j in range(channels):
                input = self.input_value[j]
                kernel = self.kernels[i][j]
                bias = self.biases[i]
                
                self.output_value[i] += Matrix.correlate(input, kernel, self.stride) + bias
    
        return self.output_value
    
    def backward(self, output_gradient: list[Matrix]) -> list[Matrix]:
        assert self.input_shape is not None
        
        channels, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        kernels_gradient: list[list[Matrix]] = [[] for _ in range(self.filters)]
        
        for filter_kernels in kernels_gradient:
            for _ in range(channels):
                kernel = Matrix(kernel_height, kernel_width)
                
                filter_kernels.append(kernel)
    
        input_gradient: list[Matrix] = [Matrix(input_height, input_width) for _ in range(channels)]
        
        for i in range(self.filters):
            for j in range(channels):
                input = self.input_value[j]
                kernel = self.kernels[i][j]
                output = output_gradient[i]
                
                kernels_gradient[i][j] = Matrix.correlate(input, output, self.stride)
                input_gradient[j] += Matrix.convolve(output, kernel, self.stride, 'full')
    
class Activation(Layer):
    ''' Abstract class for activation layers '''
    
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
    