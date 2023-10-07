from typing import Optional, Literal
from math import floor, ceil

from optimizers import Optimizer
from matrix import Matrix

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
    
    def forward(self, input_value: Matrix, training: bool = True) -> Matrix: 
        ''' Propagates input forward '''
        
        raise NotImplementedError()
    
    def backward(self, output_gradient: Matrix, training: bool = True) -> Matrix:
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
        
    def forward(self, input_value, training=True):
        self.input_value = input_value
        return self.input_value @ self.weights + self.bias
    
    def backward(self, output_gradient, training=True):
        input_gradient = output_gradient @ self.weights.T
        
        if training:
            weights_gradient = self.input_value.T @ output_gradient
            bias_gradient = output_gradient.sum_by_axis(0)
            
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
        _, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        vertical_padding, horizontal_padding = self.calculate_padding()
        
        output_height = (input_height - kernel_height + sum(vertical_padding)) // self.stride + 1
        output_width = (input_width - kernel_width + sum(horizontal_padding)) // self.stride + 1
    
        return self.filters, output_height, output_width
    
    def forward(self, input_value, training=True):
        self.input_value = input_value.pad2D(self.calculate_padding())
        
        batch_size, channels, _, _ = self.input_value.shape
        
        output_value = Matrix(batch_size, *self.output_shape)

        for b in range(batch_size):
            for f in range(self.filters):
                for c in range(channels):
                    input = self.input_value[b, c]
                    kernel = self.kernels[f, c]

                    output_value[b, f] += input.correlate2D(kernel, self.stride) + self.bias[f]

        return output_value

    def backward(self, output_gradient, training=True):
        input_gradient = Matrix(*self.input_value.shape)
        
        batch_size, channels, _, _ = self.input_value.shape
        
        if training:
            kernels_gradient = Matrix(*self.kernels.shape)
            bias_gradient = Matrix(*self.bias.shape)
        
            kernel_height, kernel_width = self.kernel_shape
        
            full_padding = ((kernel_height - 1, ) * 2, (kernel_width - 1, ) * 2)    
        
        for b in range(batch_size):
            for f in range(self.filters):
                output = output_gradient[b, f]
                
                if training:
                    bias_gradient[f] += output.sum()
                
                for c in range(channels):
                    if training:
                        input = self.input_value[b, c]
                        kernels_gradient[f, c] += input.correlate2D(output, self.stride)
                        
                    kernel = self.kernels[f, c]
                    input_gradient[b, c] += output.pad2D(full_padding).convolve2D(kernel, self.stride)

        if training:
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
        _, output_height, output_width = self.output_shape
        
        matrix = Matrix(output_height, output_width)
        
        for i in range(output_height):
            row = i * self.stride
            
            for j in range(output_width):
                col = j * self.stride
                
                matrix[i, j] = input[row: row + pool_height, col: col + pool_width].max()
                
        return matrix
    
    def forward(self, input_value, training=True):
        self.input_value = input_value.pad2D(self.calculate_padding())
        
        batch_size, channels, _, _ = self.input_value.shape
        
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
    
    def backward(self, output_gradient, training=True):
        input_gradient = Matrix(*self.input_value.shape)
        
        batch_size, channels, _, _ = self.input_value.shape
        
        for b in range(batch_size):
            for c in range(channels):
                input_gradient[b, c] = self.pool_backward(output_gradient[b, c], self.input_value[b, c])
                
        return input_gradient.crop2D(self.calculate_padding())

class UpSampling2D(Layer):
    ''' 2D up sampling layer '''

    def __init__(
        self, 
        size: tuple[int, int] = (2, 2), 
        input_shape: Optional[tuple[int, int, int]] = None
    ):
        self.size = size
        
        if input_shape is not None:
            self.set_input_shape(input_shape)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        channels, input_height, input_width = self.input_shape
        size_height, size_width = self.size
        
        height = input_height * size_height
        width = input_width * size_width
        
        return (channels, height, width)

    def forward(self, input_value, training=True):
        return input_value.repeat2D(self.size)
    
    def backward(self, output_gradient, training=True):
        batch_size, channels, _, _ = output_gradient.shape
        _, input_height, input_width = self.input_shape
        
        input_gradient = Matrix(batch_size, *self.input_shape)
        
        size_height, size_width = self.size
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(input_height):
                    for j in range(input_width):
                        output = output_gradient[b, c]
                        
                        x = i * size_height
                        y = j * size_width
                        
                        selected = output[x : x + size_height, y : y + size_width]
                        
                        input_gradient[b, c, i, j] = selected.sum()

        return input_gradient

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
    
    def forward(self, input_value, training=True):
        batch_size = input_value.shape[0]
        return input_value.reshape(batch_size, *self.output_shape)
    
    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]
        return output_gradient.reshape(batch_size, *self.input_shape)

class Reshape(Layer):
    ''' Reshape layer '''
    
    def __init__(self, shape: tuple[int, ...], input_shape: Optional[tuple[int, ...]] = None) -> None:
        self.shape = shape
        
        if input_shape is not None: 
            self.set_input_shape(input_shape)
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.shape
    
    def forward(self, input_value, training=True):
        batch_size = input_value.shape[0]
        return input_value.reshape(batch_size, *self.shape)
    
    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]
        return output_gradient.reshape(batch_size, *self.input_shape)

class Dropout(Layer):
    ''' Dropout layer '''
    
    def __init__(self, rate: float = 0.2) -> None:
        self.rate = rate
        self.scale = 1 / (1 - self.rate)
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.input_shape
    
    def forward(self, input_value, training=True):
        if training:
            self.mask = self.scale * (input_value.randomize(0, 1) > self.rate)
            
            return input_value * self.mask
        else:
            return input_value
    
    def backward(self, output_gradient, training=True):
        return output_gradient * self.mask

class BatchNormalization(Layer):
    ''' Batch normalization layer '''
    
    def __init__(self, momentum: float = 0.99, epsilon: float = 0.01) -> None:
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.running_mean: Optional[Matrix] = None
        self.running_var: Optional[Matrix] = None
        
    def initialize(self, optimizer: Optimizer) -> None:
        self.gamma = Matrix(*self.input_shape).ones()
        self.beta = Matrix(*self.input_shape).zeros()
        
        self.gamma_optimizer = optimizer.copy()
        self.beta_optimizer = optimizer.copy()
        
    def parameters(self):
        return self.gamma.count() + self.beta.count()
    
    @property
    def output_shape(self):
        return self.input_shape
    
    def forward(self, input_value, training=True):
        if self.running_mean is None or self.running_var is None:
            self.running_mean = input_value.mean_by_axis(0)
            self.running_var = input_value.var_by_axis(0)

        if training and self.running_mean is not None and self.running_var is not None:
            mean = input_value.mean_by_axis(0)
            var = input_value.var_by_axis(0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.input_centered = input_value - mean
        self.std = (var + self.epsilon) ** 0.5
        
        input_normalized = self.input_centered / self.std
        
        return self.gamma * input_normalized + self.beta
    
    def backward(self, output_gradient, training=True):
        gamma = self.gamma
        
        if training:
            input_normalized = self.input_centered / self.std
            
            gamma_gradient = (output_gradient * input_normalized).sum_by_axis(0)
            beta_gradient = output_gradient.sum_by_axis(0)
            
            self.gamma = self.gamma_optimizer.update(self.gamma, gamma_gradient)
            self.beta = self.beta_optimizer.update(self.beta, beta_gradient)
        
        batch_size = output_gradient.shape[0]
        
        return (gamma / (batch_size * self.std)) * (
            batch_size * output_gradient 
            - output_gradient.sum_by_axis(0) 
            - (self.input_centered / self.std ** 2) 
            * (output_gradient * self.input_centered).sum_by_axis(0)
        )
        
class Activation(Layer):
    ''' Activation layer '''
    
    def __call__(self, x: Matrix) -> Matrix:
        ''' Return the activation of the matrix '''
        
        raise NotImplementedError()
    
    def gradient(self, x: Matrix) -> Matrix:
        ''' Return the gradient of the matrix '''
    
        raise NotImplementedError()
    
    @property
    def output_shape(self):
        return self.input_shape
    
    def forward(self, input_value, training=True):
        self.input_value = input_value
        return self(input_value)
    
    def backward(self, output_gradient, training=True):
        return output_gradient * self.gradient(self.input_value)