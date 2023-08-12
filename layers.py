from typing import Literal, Any

from math import floor, ceil

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
    
    def forward(self, input_value: Any) -> Any: 
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
    
    def kernels_to_matrix(self, kernels: list[list[Matrix]]) -> Matrix:
        ''' Converts the kernels to a matrix '''

        matrices: list[Matrix] = []
        
        for filter_kernels in kernels:
            for kernel in filter_kernels:
                matrices.append(kernel)

        return Matrix.join(matrices)
    
    def matrix_to_kernels(self, matrix: Matrix) -> list[list[Matrix]]:
        ''' Converts the matrix to kernels '''
        
        assert self.input_shape is not None
        
        channels, _, _ = self.input_shape
        kernel_height, _ = self.kernel_shape
        
        kernels: list[list[Matrix]] = [[] for _ in range(self.filters)]
        
        for f in range(self.filters):
            for c in range(channels):
                index = (f * channels + c) * kernel_height
                
                kernels[f].append(matrix[index:index + kernel_height, :])
        
        return kernels
    
    def create_kernels(self):
        ''' Creates the kernels '''
        
        assert self.input_shape is not None
        
        channels, _, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_shape
        
        kernels: list[list[Matrix]] = [[] for _ in range(self.filters)]
        
        for filter_kernels in self.kernels:
            for _ in range(channels):
                filter_kernels.append(Matrix(kernel_height, kernel_width))
    
        return kernels
    
    def get_padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        ''' Returns the padding size '''
    
        kernel_height, kernel_width = self.kernel_shape
        
        if self.padding == 'valid':
            return ((0, 0), (0, 0))

        v_pad = ((kernel_height - 1) // 2, ceil((kernel_height - 1) / 2))
        h_pad = ((kernel_width - 1) // 2, ceil((kernel_width - 1) / 2))
    
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
        
        self.kernels = self.create_kernels()
        
        limit = 1 / (kernel_height * kernel_width) ** 0.5
        
        for f in range(self.filters):
            for c in range(channels):
                self.kernels[f][c] = self.kernels[f][c].randomize(-limit, limit)
        
        self.biases: list[float] = [0] * self.filters
        
        self.kernels_optimizer = optimizer.copy()
        self.biases_optimizer = optimizer.copy()
    
    def forward(self, input_value: list[Matrix]) -> list[Matrix]:
        assert self.input_shape is not None
        
        v_pad, h_pad = self.get_padding()
        self.input_value = [channel.pad(v_pad, h_pad) for channel in input_value]
        
        _, output_height, output_width = self.output_shape()
        self.output_value = [Matrix(output_height, output_width) for i in range(self.filters)]
        
        channels, _, _ = self.input_shape
        
        for f in range(self.filters):
            self.output_value[f] += self.biases[f]
            
            for c in range(channels):
                input = self.input_value[c]
                kernel = self.kernels[f][c]
                
                self.output_value[f] += Matrix.correlate(input, kernel, self.stride)
    
        return self.output_value
    
    def backward(self, output_gradient: list[Matrix]) -> list[Matrix]:
        assert self.input_shape is not None
        
        channels, input_height, input_width = self.input_shape
        
        kernels_gradient = self.create_kernels()
        
        input_gradient = [Matrix(input_height, input_width) for _ in range(channels)]
        biases_gradient = [output_gradient[i].sum() for i in range(self.filters)]
        
        for f in range(self.filters):
            for c in range(channels):
                input = self.input_value[c]
                kernel = self.kernels[f][c]
                output = output_gradient[f]
                
                kernels_gradient[f][c] = Matrix.correlate(input, output, self.stride)
                
                v_pad = (kernel.rows - 1, kernel.rows - 1)
                h_pad = (kernel.columns - 1, kernel.columns - 1)
                
                input_gradient[c] += Matrix.convolve(output, kernel.pad(v_pad, h_pad), self.stride)
    
        kernels_update = self.kernels_optimizer.update(
            self.kernels_to_matrix(self.kernels),
            self.kernels_to_matrix(kernels_gradient)
        )
        
        self.kernels = self.matrix_to_kernels(kernels_update)
    
        biases_update = self.biases_optimizer.update(
            Matrix.from_list(self.biases),
            Matrix.from_list(biases_gradient)
        )
        
        self.biases = biases_update.to_list()
        
        return [input_gradient[i].crop(*self.get_padding()) for i in range(channels)]
    
class MaxPooling2D(Layer):
    ''' Max pooling layer '''
    
    def __init__(
        self, 
        pool_shape: tuple[int, int], 
        stride: int = 1, 
        padding: Literal['valid', 'same'] = 'valid'
    ):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

    def get_padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        ''' Returns the padding size '''

        pool_height, pool_width = self.pool_shape
        
        if self.padding == 'valid':
            return ((0, 0), (0, 0))
        
        v_pad = ((pool_height - 1) // 2, ceil((pool_height - 1) / 2))
        h_pad = ((pool_width - 1) // 2, ceil((pool_width - 1) / 2))

        return (v_pad, h_pad)
    
    def output_shape(self) -> tuple[int, ...]:
        assert self.input_shape is not None
        
        channels, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pool_shape
        
        match self.padding:
            case 'valid':
                height = (input_height - pool_height) // self.stride + 1
                width = (input_width - pool_width) // self.stride + 1
            case 'same':
                height = (input_height - 1) // self.stride + 1
                width = (input_width - 1) // self.stride + 1
        
        return (channels, height, width)

    def forward(self, input_value: list[Matrix]) -> list[Matrix]:
        v_pad, h_pad = self.get_padding()
        
        self.input_value = [channel.pad(v_pad, h_pad) for channel in input_value]
    
        channels, output_height, output_width = self.output_shape()
        self.output_value = [Matrix(output_height, output_width) for _ in range(channels)]
        
        pool_height, pool_width = self.pool_shape
        
        self.indexes: list[list[tuple[int, int]]] = [[] for _ in range(channels)]
        
        for i in range(channels):
            input = self.input_value[i]
            
            for r in range(output_height):
                row = r * self.stride
                
                for c in range(output_width):
                    col = c * self.stride
                    
                    input_part = input[row : row + pool_height, col : col + pool_width]
                    
                    self.output_value[i].data[r][c] = input_part.max()
                    
                    row_idx, col_idx = input_part.index(input_part.max())
                    self.indexes[i].append((row_idx + row, col_idx + col))
    
        return self.output_value
    
    def backward(self, output_gradient: list[Matrix]) -> list[Matrix]:
        assert self.input_shape is not None
        
        channels, input_height, input_width = self.input_shape
        input_gradient = [Matrix(input_height, input_width) for _ in range(channels)]
        
        print(self.indexes)
        
        for i in range(channels):
            output = output_gradient[i]
            input = input_gradient[i]
            
            for r in range(output.rows):
                for c in range(output.cols):
                    row, col = self.indexes[i][r * output.cols + c]
                    input.data[row][col] = output.data[r][c]

        return input_gradient

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
    