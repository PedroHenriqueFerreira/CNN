from typing import Optional, Literal, Any

from math import floor, ceil

from matrix import Matrix
from optimizers import Optimizer, AdamOptimizer

class Layer:
    ''' Abstract class for layers '''
    
    def set_input_shape(self, input_shape: tuple[int, ...]) -> None:
        ''' Sets the layer input shape '''
        
        self.input_shape = input_shape
    
    def initialize(self, optimizer: Optimizer) -> None:
        ''' Initialize the layer '''
        
        return None
    
    def parameters(self) -> int:
        ''' Returns the number of parameters in the layer '''
        
        return 0
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        ''' Returns the layer output shape '''
        
        raise NotImplementedError()
    
    def forward(self, input_value: Matrix) -> Matrix: 
        ''' Propagates input forward '''
        
        raise NotImplementedError()
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        ''' Propagates gradient backward '''
        
        raise NotImplementedError()
    
class Dense(Layer):
    ''' Fully connected layer '''
    
    def __init__(self, units: int, input_shape: Optional[tuple[int]] = None):
        ''' Initialize a fully connected layer '''
        
        self.units = units
        
        if input_shape is not None: 
            self.set_input_shape(input_shape)
    
    def initialize(self, optimizer: Optimizer) -> None:
        limit = 1 / (self.input_shape[0] ** 0.5)
        
        self.weights = Matrix(self.input_shape[0], self.units).randomize(-limit, limit)
        self.bias = Matrix(1, self.units).zeros()
        
        self.weights_optimizer = optimizer.copy()
        self.bias_optimizer = optimizer.copy()
    
    def parameters(self) -> int:
        return self.weights.count() + self.bias.count() # type: ignore
        
    @property
    def output_shape(self) -> tuple[int]:
        return (self.units, )
        
    def forward(self, input_value: Matrix) -> Matrix:
        self.input_value = input_value

        return self.input_value @ self.weights + self.bias
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        weights_gradient = self.input_value.T @ output_gradient
        bias_gradient = output_gradient.sum_by_axis(0)
        input_gradient = output_gradient @ self.weights.T
        
        self.weights = self.weights_optimizer.update(self.weights, weights_gradient)
        self.bias = self.bias_optimizer.update(self.bias, bias_gradient)
        
        return input_gradient
    
class Conv2D(Layer):
    ''' 2D convolution layer '''
    
    def __init__(
        self, 
        filters: int, 
        kernel_shape: tuple[int, int], 
        input_shape: Optional[tuple[int, int, int]] = None, 
        stride: int = 1,
        padding: Literal['valid', 'same'] = 'valid'
    ):
    
        self.filters = filters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        
        if input_shape is not None: 
            self.set_input_shape(input_shape)
            
    def initialize(self, optimizer: Optimizer) -> None:
        channels = self.input_shape[0]
        kernel_height, kernel_width = self.kernel_shape
        
        limit = 1 / (kernel_height * kernel_width) ** 0.5
        
        self.kernels = Matrix(self.filters, channels, kernel_height, kernel_width).randomize(-limit, limit)
        self.bias = Matrix(self.filters, 1).zeros()

        self.kernels_optimizer = optimizer.copy()
        self.bias_optimizer = optimizer.copy()
        
    def parameters(self) -> int:
        return self.kernels.count() + self.bias.count() # type: ignore
    
    def calculate_padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        ''' Calculates the padding for the convolution '''
        
        kernel_height, kernel_width = self.kernel_shape
        
        if self.padding == 'valid':
            return ((0, 0), (0, 0))
        
        vertical = floor((kernel_height - 1) / 2), ceil((kernel_height - 1) / 2)
        horizontal = floor((kernel_width - 1) / 2), ceil((kernel_width - 1) / 2)
        
        return vertical, horizontal
    
    @property
    def output_shape(self) -> tuple[int, int, int]:
        channels, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        vertical_padding, horizontal_padding = self.calculate_padding()
        
        output_height = (input_height - kernel_height + sum(vertical_padding)) // self.stride + 1
        output_width = (input_width - kernel_width + sum(horizontal_padding)) // self.stride + 1
    
        return self.filters, output_height, output_width
    
    def forward(self, input_value: Matrix) -> Any:
        self.input_value = input_value.pad2D(self.calculate_padding())
        
        batch_size, channels, input_height, input_width = self.input_value.shape
        
        output_value = Matrix(batch_size, *self.output_shape)

        for b in range(batch_size):
            for f in range(self.filters):
                for c in range(channels):
                    input = self.input_value[b, c]
                    kernel = self.kernels[f, c]

                    output_value[b, f] += input.correlate2D(kernel, self.stride) + self.bias[f]

        return output_value

    def backward(self, output_gradient: Matrix) -> Matrix:
        kernels_gradient = Matrix(*self.kernels.shape)
        input_gradient = Matrix(*self.input_value.shape)
        bias_gradient = Matrix(*self.bias.shape)
        
        batch_size, channels, input_height, input_width = self.input_value.shape
        kernel_height, kernel_width = self.kernel_shape
        
        full_padding = ((kernel_height - 1, ) * 2, (kernel_width - 1, ) * 2)
        
        for b in range(batch_size):
            for f in range(self.filters):
                output = output_gradient[b, f]
                
                bias_gradient[f] += output.sum()
                
                for c in range(channels):
                    input = self.input_value[b, c]
                    kernel = self.kernels[f, c]
                    
                    kernels_gradient[f, c] += input.correlate2D(output, self.stride)
                    input_gradient[b, c] += output.pad2D(full_padding).convolve2D(kernel, self.stride)

        self.kernels = self.kernels_optimizer.update(self.kernels, kernels_gradient)
        self.bias = self.bias_optimizer.update(self.bias, bias_gradient)

        return input_gradient.crop2D(self.calculate_padding())

class MaxPooling2D(Layer):
    ''' 2D max pooling layer '''
    
    def __init__(
        self, 
        pool_shape: tuple[int, int], 
        stride: int = 1, 
        padding: Literal['valid', 'same'] = 'valid'
    ):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        
    def calculate_padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        ''' Calculates the padding for the pooling '''
        
        if self.padding == 'valid':
            return ((0, 0), (0, 0))
        
        pool_height, pool_width = self.pool_shape
        
        vertical = floor((pool_height - 1) / 2), ceil((pool_height - 1) / 2)
        horizontal = floor((pool_width - 1) / 2), ceil((pool_width - 1) / 2)
        
        return vertical, horizontal
    
    @property
    def output_shape(self) -> tuple[int, int, int]:
        channels, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pool_shape
        
        vertical_padding, horizontal_padding = self.calculate_padding()
        
        output_height = (input_height - pool_height + sum(vertical_padding)) // self.stride + 1
        output_width = (input_width - pool_width + sum(horizontal_padding)) // self.stride + 1
    
        return channels, output_height, output_width
    
    def pool_forward(self, input: Matrix) -> Matrix:
        pool_height, pool_width = self.pool_shape
        channels, output_height, output_width = self.output_shape
        
        matrix = Matrix(output_height, output_width)
        
        for i in range(output_height):
            row = i * self.stride
            
            for j in range(output_width):
                col = j * self.stride
                
                matrix[i, j] = input[row: row + pool_height, col: col + pool_width].max()
                
        return matrix
    
    def forward(self, input_value: Matrix) -> Matrix:
        self.input_value = input_value.pad2D(self.calculate_padding())
        
        batch_size, channels, input_height, input_width = self.input_value.shape
        
        output_value = Matrix(batch_size, *self.output_shape)
        
        for b in range(batch_size):
            for c in range(channels):
                output_value[b, c] = self.pool_forward(self.input_value[b, c])
                
        return output_value
    
    def pool_backward(self, output: Matrix, input: Matrix) -> Matrix:
        matrix = Matrix(*input.shape)
        
        pool_height, pool_width = self.pool_shape
        output_height, output_width = output.shape
        
        for i in range(output_height):
            row = i * self.stride
            
            for j in range(output_width):
                col = j * self.stride
                
                selected = input[row: row + pool_height, col: col + pool_width]
                value = selected.max()
                
                for k in range(pool_height):
                    should_break = False
                
                    for l in range(pool_width):
                        if selected[k : k + 1, l : l + 1] == value:
                            matrix[row + k, col + l] = output[i, j]
                            should_break = True
                            break
                    
                    if should_break:
                        break
                
        return matrix
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        input_gradient = Matrix(*self.input_value.shape)
        
        batch_size, channels, input_height, input_width = self.input_value.shape
        
        for b in range(batch_size):
            for c in range(channels):
                input_gradient[b, c] = self.pool_backward(output_gradient[b, c], self.input_value[b, c])
                
        return input_gradient.crop2D(self.calculate_padding())

class Flatten(Layer):
    ''' Flatten layer '''
    
    def __init__(self, input_shape: Optional[tuple[int, ...]] = None):
        
        if input_shape is not None: 
            self.set_input_shape(input_shape)
            
    @property
    def output_shape(self) -> tuple[int]:
        result = 1
        
        for dim in self.input_shape:
            result *= dim
            
        return (result, )
    
    def forward(self, input_value: Matrix) -> Matrix:
        self.batch_size = input_value.shape[0]
    
        return input_value.reshape(self.batch_size, *self.output_shape)
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        return output_gradient.reshape(self.batch_size, *self.input_shape)

class Reshape(Layer):
    ''' Reshape layer '''
    
    def __init__(self, shape: tuple[int, ...], input_shape: Optional[tuple[int, ...]] = None) -> None:
        self.shape = shape
        
        if input_shape is not None: 
            self.set_input_shape(input_shape)
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.shape
    
    def forward(self, input_value: Matrix) -> Matrix:
        self.batch_size = input_value.shape[0]
        
        return input_value.reshape(self.batch_size, *self.shape)
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        return output_gradient.reshape(self.batch_size, *self.input_shape)

class Dropout(Layer):
    ''' Dropout layer '''
    
    def __init__(self, rate: float = 0.2) -> None:
        self.rate = rate
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.input_shape
    
    def forward(self, input_value: Matrix) -> Matrix:
        self.mask = input_value.randomize(0, 1) > self.rate
        
        return input_value * self.mask
    
    def backward(self, output_gradient: Matrix) -> Matrix:
        return output_gradient * self.mask
