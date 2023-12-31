from typing import Union, Callable, Any
from pprint import pformat

from math import log, exp, inf
from random import uniform

from PIL import Image

class Matrix:
    def __init__(self, *shape: int):
        ''' Initialize a new matrix with the given shape '''
        
        self.shape = shape
        
        self.data: list[Any] = self.create_data(self.shape)
       
    def copy(self) -> 'Matrix':
        ''' Return a copy of the matrix '''
        
        self_arr = self.to_array()
        
        return self.map(lambda _: self_arr.pop(0))
        
    def create_data(self, shape: tuple[int, ...]) -> list[Any]:
        ''' Create a list of the given shape recursively '''
        
        if len(shape) > 0:
            return [self.create_data(shape[1:]) for _ in range(shape[0])]
        
        return 0 # type: ignore
    
    def map_data(self, data: list[Any], function: Callable[[float], float]) -> list[Any]:
        ''' Map a function to the data recursively '''
        
        if isinstance(data, list):
            return [self.map_data(items, function) for items in data]
        
        try:
            return function(data)
        except OverflowError:
            return inf
    
    def array_data(self, data: list[Any]) -> list[float]:
        ''' Flatten the data recursively '''
        
        if isinstance(data, list) and isinstance(data[0], list):
            return [item for items in data for item in self.array_data(items)]
        
        return data
    
    def select_data(self, data: list[Any], indexes: tuple[slice | int, ...]) -> list[Any]:
        ''' Filter the data recursively '''
        
        if len(indexes) > 0:
            if isinstance(indexes[0], slice):
                return [self.select_data(items, indexes[1:]) for items in data[indexes[0]]]
            elif isinstance(indexes[0], int):
                return self.select_data(data[indexes[0]], indexes[1:])
    
        return data
    
    def set_data(self, data: list[Any], key: tuple[int, ...], value: list[Any] | float) -> None:
        ''' Set the value at the given key recursively '''
        
        if len(key) > 1:
            self.set_data(data[key[0]], key[1:], value)
        else:
            data[key[0]] = value
    
    def repeat2D_data(self, data: list[Any], repeats: tuple[int, int]) -> list[Any]:
        ''' Repeat the 2D matrix data with the given repeats '''
        
        if isinstance(data, list):
            if isinstance(data[0], list) and isinstance(data[0][0], float | int):
                height = len(data) * repeats[0]
                width = len(data[0]) * repeats[1]
                
                matrix = [[0] * width for _ in range(height)]
                
                for i in range(len(data)):
                    for j in range(len(data[0])):
                        for k in range(repeats[0]):
                            for l in range(repeats[1]):
                                matrix[i * repeats[0] + k][j * repeats[1] + l] = data[i][j]
                
                return matrix

            return [self.repeat2D_data(items, repeats) for items in data]

        return data
    
    def pad2D_data(self, data: list[Any], padding: tuple[tuple[int, int], tuple[int, int]]) -> list[Any]:
        ''' Pad the 2D matrix data with the given padding '''
        
        if isinstance(data, list):
            if isinstance(data[0], list) and isinstance(data[0][0], float | int):
                row_size = len(data[0]) + sum(padding[1])
                
                top = [[0] * row_size for _ in range(padding[0][0])]
                bottom = [[0] * row_size for _ in range(padding[0][1])]
                
                data = [[0] * padding[1][0] + items + [0] * padding[1][1] for items in data]
                
                return top + data + bottom

            return [self.pad2D_data(items, padding) for items in data]

        return data
    
    def crop2D_data(self, data: list[Any], cropping: tuple[tuple[int, int], tuple[int, int]]) -> list[Any]:
        ''' Crop the 2D matrix data with the given cropping '''
        
        if isinstance(data, list):
            if isinstance(data[0], list) and isinstance(data[0][0], float | int):
                rows = len(data) - sum(cropping[0])
                cols = len(data[0]) - sum(cropping[1])
                
                matrix = [[0] * cols for _ in range(rows)]
                
                for i in range(rows):
                    for j in range(cols):
                        matrix[i][j] = data[i + cropping[0][0]][j + cropping[1][0]]

                return matrix

            return [self.crop2D_data(items, cropping) for items in data]

        return data
    
    def split(self, size: int) -> list['Matrix']:
        ''' Split the matrix into a list of matrices of the given size '''
        
        parts = self.shape[0] / size
        
        if int(parts) < parts:
            parts = int(parts) + 1
        
        return [self[i * size : (i + 1) * size] for i in range(int(parts))]
    
    def map(self, function: Callable[[float], float]) -> 'Matrix':
        ''' Map a function to the matrix '''
        
        return Matrix.load(self.map_data(self.data, function))
    
    def zeros(self) -> 'Matrix':
        ''' Fill the matrix with zeros '''
        
        return self.map(lambda _: 0)
    
    def ones(self) -> 'Matrix':
        ''' Fill the matrix with ones '''
        
        return self.map(lambda _: 1)
  
    def randomize(self, min_value: float = -1, max_value: float = 1) -> 'Matrix':
        ''' Randomize the matrix with values between min_value and max_value '''
        
        return self.map(lambda _: uniform(min_value, max_value))
    
    def clip(self, min_val: float, max_val: float) -> 'Matrix':
        ''' Clip all values in the matrix between min and max '''
        
        return self.map(lambda x: min(max(x, min_val), max_val))
    
    def log(self) -> 'Matrix':
        ''' Take the logarithm of all values in the matrix '''
        
        return self.map(lambda x: log(x))
    
    def exp(self) -> 'Matrix':
        ''' Take the exponential of all values in the matrix '''
        
        return self.map(lambda x: exp(x))
    
    def sqrt(self) -> 'Matrix':
        ''' Take the square root of all values in the matrix '''
        
        return self.map(lambda x: x ** 0.5) # type: ignore
 
    def max(self):
        ''' Calculate the maximum of all values in the matrix '''
        
        return max(self.to_array())
    
    def sum(self) -> float:
        ''' Calculate the sum of all values in the matrix '''
        
        return sum(self.to_array())
    
    def count(self):
        ''' Calculate the number of values in the matrix '''
        
        return len(self.to_array())
    
    def mean(self):
        ''' Calculate the mean of all values in the matrix '''
        
        if self.count() == 0:
            return 0.0
        
        return self.sum() / self.count()

    def var(self):
        ''' Calculate the variance of all values in the matrix '''
        
        if self.count() == 0:
            return 0.0
        
        mean = self.mean()
        
        return self.map(lambda x: (x - mean) ** 2).mean() # type: ignore

    def to_array(self) -> list[float]:
        ''' Convert the matrix to a list '''
        
        return self.array_data(self.data)

    def reshape(self, *shape: int) -> 'Matrix':
        ''' Reshape the matrix '''        
        
        matrix = Matrix(*shape)
        
        if self.count() != matrix.count():
            raise ValueError('Invalid reshape dimensions')
        
        self_arr = self.to_array()
        
        return matrix.map(lambda _: self_arr.pop(0))

    def sum_by_axis(self, axis: int) -> 'Matrix':
        ''' Sum all values in the matrix along the given axis '''

        matrix = Matrix(*self.shape[1:axis], 1, *self.shape[axis + 1:])
        
        for i in range(self.shape[0]):
            slices = (slice(None),) * axis
            
            matrix += self[*slices, i : i + 1]
        
        return matrix

    def mean_by_axis(self, axis: int) -> 'Matrix':
        ''' Calculate the mean of all values in the matrix along the given axis '''
            
        return self.sum_by_axis(axis) / self.shape[axis]

    def var_by_axis(self, axis: int) -> 'Matrix':
        ''' Calculate the variance of all values in the matrix along the given axis '''
        
        mean = self.mean_by_axis(axis)
        
        matrix = Matrix(*self.shape[1:axis], 1, *self.shape[axis + 1:])
        
        for i in range(self.shape[0]):
            slices = (slice(None),) * axis
                    
            matrix += (self[*slices, i : i + 1] - mean) ** 2
        
        return matrix / self.shape[axis]

    def operation(self, other: 'Matrix', op: Callable[[float, float], float]) -> 'Matrix':
        ''' Perform an operation on two matrices '''
        
        self_arr = self.to_array()
        other_arr = other.to_array()
        
        if len(self_arr) == len(other_arr):
            if (sum(self.shape) >= sum(other.shape)):
                larger = self_arr
                smaller = other_arr
            else:
                larger = other_arr
                smaller = self_arr
            
        else:
            larger = max(self_arr, other_arr, key=len)
            smaller = min(self_arr, other_arr, key=len)
            
            times = len(larger) // len(smaller)
            remaining = len(larger) % len(smaller)
            
            smaller = smaller * times + smaller[:remaining]
        
        if larger is self_arr:
            return self.map(lambda _: op(larger.pop(0), smaller.pop(0)))
        else:
            return other.map(lambda _: op(smaller.pop(0), larger.pop(0)))

    def __neg__(self) -> 'Matrix':
        ''' Negate the matrix values '''
        
        return self.map(lambda x: -x)
    
    def __add__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Add two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value + other) # type: ignore
        
        return self.operation(other, lambda value, other_value: value + other_value)

    def __radd__(self, other: float) -> 'Matrix':
        ''' Add a scalar and a matrix '''
        
        return self.map(lambda value: other + value)

    def __sub__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Subtract two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value - other) # type: ignore
        
        return self.operation(other, lambda value, other_value: value - other_value)

    def __rsub__(self, other: float) -> 'Matrix':
        ''' Subtract a scalar and a matrix '''
        
        return self.map(lambda value: other - value)
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Multiply two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value * other) # type: ignore
        
        return self.operation(other, lambda value, other_value: value * other_value)

    def __rmul__(self, other: float) -> 'Matrix':
        ''' Multiply a scalar and a matrix '''
        
        return self.map(lambda value: other * value)
    
    def __truediv__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Divide two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value / other) # type: ignore
        
        return self.operation(other, lambda value, other_value: value / other_value)

    def __rtruediv__(self, other: float) -> 'Matrix':
        ''' Divide a scalar and a matrix '''
        
        return self.map(lambda value: other / value)
    
    def __pow__(self, other: float) -> 'Matrix':
        ''' Raise a matrix to a power '''
        
        return self.map(lambda value: value ** other) # type: ignore
    
    def __rpow__(self, other: float) -> 'Matrix':
        ''' Raise a scalar to a power '''
        
        return self.map(lambda value: other ** value) # type: ignore
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        ''' Dot product of two 2D matrices '''
    
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError('Matrices must be 2D')
        
        if self.shape[1] != other.shape[0]:
            raise ValueError('Matrices must have compatible dimensions')
        
        matrix = Matrix(self.shape[0], other.shape[1])    
    
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    matrix.data[i][j] += self.data[i][k] * other.data[k][j]
    
        return matrix
    
    def __getitem__(self, key: tuple[slice | int, ...] | slice | int) -> 'Matrix':
        ''' Filter the matrix with the given key '''
        
        if isinstance(key, slice | int):
            key = (key,)
        
        return Matrix.load(self.select_data(self.data, key))

    def __setitem__(self, key: tuple[int, ...] | int, value: Union['Matrix', list[Any], float]) -> None:
        ''' Set the value by the given key '''
        
        if isinstance(key, int):
            key = (key,)
        
        if isinstance(value, list):
            value = Matrix.load(value)
        
        if isinstance(value, Matrix):
            if len(key) + len(value.shape) != len(self.shape):
                raise ValueError('Invalid dimensions')
            
            value = value.data
            
        if isinstance(value, float | int):
            if len(key) != len(self.shape):
                raise ValueError('Invalid dimensions')
        
        self.set_data(self.data, key, value)

    def pad2D(self, padding: tuple[tuple[int, int], tuple[int, int]]) -> 'Matrix':
        ''' Pad the 2D matrices inside matrix with the given padding '''
        
        return Matrix.load(self.pad2D_data(self.data, padding))
    
    def crop2D(self, cropping: tuple[tuple[int, int], tuple[int, int]]) -> 'Matrix':
        ''' Crop the 2D matrices inside matrix with the given cropping '''
        
        return Matrix.load(self.crop2D_data(self.data, cropping))
    
    def repeat2D(self, repeats: tuple[int, int]) -> 'Matrix':
        ''' Repeat the 2D matrices inside matrix with the given repeats '''
        
        return Matrix.load(self.repeat2D_data(self.data, repeats))
    
    def correlate2D(self, kernel: 'Matrix', stride: int = 1) -> 'Matrix':
        ''' Correlate the 2D matrix with the given kernel and stride '''
        
        if len(self.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError('Matrices must be 2D')
        
        kernel_rows, kernel_cols = kernel.shape
        
        rows = (self.shape[0] - kernel_rows) // stride + 1
        cols = (self.shape[1] - kernel_cols) // stride + 1
        
        matrix = Matrix(rows, cols)
        
        for i in range(rows):
            row = i * stride
            
            for j in range(cols):
                col = j * stride
                
                selected = self[row:row + kernel_rows, col:col + kernel_cols]
                
                matrix[i, j] = (selected * kernel).sum()
    
        return matrix
    
    def convolve2D(self, kernel: 'Matrix', stride: int = 1) -> 'Matrix':
        ''' Convolve the 2D matrix with the given kernel and stride '''
        
        return self.correlate2D(kernel.Rot180, stride)
    
    @property
    def T(self) -> 'Matrix':
        ''' Return the 2D matrix transposed '''
        
        if len(self.shape) != 2:
            raise ValueError('Matrix must be 2D')
        
        rows, cols = self.shape
        
        matrix = Matrix(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                matrix.data[j][i] = self.data[i][j]
        
        return matrix
    
    @property
    def Rot180(self) -> 'Matrix':
        ''' Return the 2D matrix rotated 180 degrees '''
        
        if len(self.shape) != 2:
            raise ValueError('Matrix must be 2D')
        
        rows, cols = self.shape
        
        matrix = Matrix(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                matrix.data[i][j] = self.data[rows - i - 1][cols - j - 1]
    
        return matrix
    
    @staticmethod
    def load(data: list[Any]) -> 'Matrix':
        ''' Load a matrix from a list '''
        
        shape: list[int] = []
        
        items = data
        
        while isinstance(items, list):
            if isinstance(items[0], list) and any(len(items[0]) != len(item) for item in items):
                raise ValueError('Inhomogeneous matrix dimensions')
            
            shape.append(len(items))
        
            items = items[0]
        
        matrix = Matrix(*shape)
        matrix.data = data
        
        return matrix
    
    @staticmethod
    def from_image(path: str) -> 'Matrix':
        image = Image.open(path).convert('RGB')
        data: list[int] = list(image.getdata())
        
        matrix = Matrix(3, image.height, image.width)
        
        for i in range(3):
            for j in range(image.height):
                for k in range(image.width):
                    matrix[i, j, k] = data[j * image.width + k][i]
        
        return matrix
    
    def to_image(self, path: str) -> None:
        image = Image.new('RGB', (self.shape[2], self.shape[1]))
        
        for j in range(self.shape[1]):
            for i in range(self.shape[2]):
                r = int(self.data[0][j][i])
                g = int(self.data[1][j][i])
                b = int(self.data[2][j][i])
                
                image.putpixel((i, j), (r, g, b))
    
        image.save(path)
    
    def __str__(self):
        ''' Return a string representation of the matrix ''' 
        
        return self.__repr__()
    
    def __repr__(self) -> str:
        ''' Return a string representation of the matrix '''
        
        return pformat(self.data)
    
    def __bool__(self) -> bool:
        ''' Check if the matrix values are not all zero '''
        
        return all(value != 0 for value in self.to_array())
    
    def __eq__(self, other: float) -> 'Matrix': # type: ignore
        ''' Check if the matrix values are equal to a scalar '''
        
        return self.map(lambda value: int(value == other))
    
    def __lt__(self, other: float) -> 'Matrix':
        ''' Check if the matrix values are less than a scalar '''
        
        return self.map(lambda value: int(value < other))
    
    def __gt__(self, other: float) -> 'Matrix':
        ''' Check if the matrix values are greater than a scalar '''
        
        return self.map(lambda value: int(value > other))