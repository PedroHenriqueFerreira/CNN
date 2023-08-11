from typing import Callable, Union, Literal
from random import uniform

class Matrix:
    ''' Matrix class for matrix operations '''
    
    def __init__(self, rows: int, cols: int):
        ''' Initialize a matrix with the given dimensions '''
        
        if rows <= 0 or cols <= 0:
            raise ValueError('Invalid matrix dimensions')
        
        self.rows = rows
        self.cols = cols
        
        self.data: list[list[float]] = [[0] * cols for _ in range(rows)]
    
    def map(self, function: Callable[[float], float]) -> 'Matrix':
        ''' Apply a function to each element in the matrix by item '''
    
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = function(self.data[i][j])
                
        return matrix
    
    def randomize(self, min_value: float = 0, max_value: float = 1) -> 'Matrix':
        ''' Randomize matrix values '''
  
        return self.map(lambda _: uniform(min_value, max_value))
    
    def zeros(self) -> 'Matrix':
        ''' Set all values in the matrix to zero '''
        
        return self.map(lambda _: 0.0)
    
    def fill(self, values: list[float]) -> 'Matrix':
        ''' Fill matrix with values from a list '''
        
        if len(values) != self.count():
            raise ValueError('Values must have the same length as the matrix')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = values[i * self.cols + j]
    
        return matrix
    
    @staticmethod  
    def load(data: list[list[float]]) -> 'Matrix':
        ''' Load a matrix from a list of lists '''
        
        if len(data) == 0:
            raise ValueError('List must contain at least one element')

        if len(data[0]) == 0:
            raise ValueError('Sub-lists must contain at least one element')
        
        if any(len(row) != len(data[0]) for row in data):
            raise ValueError('Sub-lists must have the same length')
        
        matrix = Matrix(len(data), len(data[0]))
        matrix.data = [row[:] for row in data]

        return matrix
    
    @staticmethod
    def from_list(values: list[float]) -> 'Matrix':
        ''' Convert a list to a matrix '''
        
        if len(values) == 0:
            raise ValueError('List must contain at least one element')
        
        return Matrix(len(values), 1).fill(values)
    
    def to_list(self) -> list[float]:
        ''' Convert matrix to a list '''
        
        return [self.data[i][j] for i in range(self.rows) for j in range(self.cols)]

    def sum(self) -> float:
        ''' Return the sum of all elements in the matrix '''
        
        return sum(self.to_list())
    
    def max(self) -> float:
        ''' Return the maximum value in the matrix '''
        
        return max(self.to_list())
    
    def count(self) -> int:
        ''' Return the number of elements in the matrix '''
        
        return self.rows * self.cols
    
    def mean(self) -> float:
        ''' Return the mean of all elements in the matrix '''
        
        if self.count() == 0:
            return 0.0
        
        return self.sum() / self.count()
    
    def __add__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Add two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value + other) # type: ignore
        
        if [self.cols, other.rows] != [other.cols, self.rows]:
            raise ValueError('Matrices must have the same dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[i][j] + other.data[i][j]
        
        return matrix
    
    def __radd__(self, other: float) -> 'Matrix':
        ''' Add a matrix and a scalar '''
        
        return self.map(lambda value: other + value)
    
    def __sub__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Subtract two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value - other) # type: ignore
        
        if [self.cols, other.rows] != [other.cols, self.rows]:
            raise ValueError('Matrices must have the same dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[i][j] - other.data[i][j]
        
        return matrix
    
    def __rsub__(self, other: float) -> 'Matrix':
        ''' Subtract matrix and a scalar '''
        
        return self.map(lambda value: other - value)
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Multiply two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value * other) # type: ignore
        
        if [self.cols, other.rows] != [other.cols, self.rows]:
            raise ValueError('Matrices must have the same dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[i][j] * other.data[i][j]
        
        return matrix
    
    def __rmul__(self, other: float) -> 'Matrix':
        ''' Multiply a matrix and a scalar '''
        
        return self.map(lambda value: other * value)
    
    def __truediv__(self, other: Union['Matrix', float]) -> 'Matrix':
        ''' Divide two matrices or a matrix and a scalar '''
        
        if isinstance(other, float | int):
            return self.map(lambda value: value / other) # type: ignore
        
        if [self.cols, other.rows] != [other.cols, self.rows]:
            raise ValueError('Matrices must have the same dimensions')
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[i][j] / other.data[i][j]
        
        return matrix
    
    def __rtruediv__(self, other: float) -> 'Matrix':
        ''' Divide a matrix and a scalar '''
        
        return self.map(lambda value: other / value)
    
    def __pow__(self, other: float) -> 'Matrix':    
        ''' Raise matrix to a power '''
        
        return self.map(lambda value: value ** other) # type: ignore
    
    def __rpow__(self, other: float) -> 'Matrix':
        ''' Raise a scalar to a power '''
        
        return self.map(lambda value: other ** value) # type: ignore
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        ''' Dot product of two matrices '''
        
        if self.cols != other.rows: 
            ValueError('Matrices must have compatible dimensions')
            
        matrix = Matrix(self.rows, other.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for k in range(self.cols):
                    matrix.data[i][j] += self.data[i][k] * other.data[k][j]

        return matrix
    
    @staticmethod
    def join(*matrices: 'Matrix') -> 'Matrix':
        ''' Join matrices along the specified axis '''
        
        if len(matrices) == 0:
            raise ValueError('At least one matrix must be provided')
        
        if len(set(matrix.cols for matrix in matrices)) > 1:
            raise ValueError('Invalid matrix dimensions')
    
        data: list[list[float]] = []
        
        for matrix in matrices:
            data.extend(matrix.data)
            
        return Matrix.load(data)
               
    @staticmethod
    def correlate(
        base: 'Matrix', 
        kernel: 'Matrix', 
        stride: int = 1, 
        mode: Literal['valid', 'same', 'full'] = 'valid'
    ) -> 'Matrix':
        ''' Correlate the matrix with a kernel '''
        
        if kernel.rows > base.rows or kernel.cols > base.cols:
            raise ValueError('Kernel must be smaller than the base matrix')
        
        match mode:
            case 'valid':
                v_pad = (0, 0)
                h_pad = (0, 0)
            
            case 'same':
                v_total_pad = (base.rows - 1) * stride - base.rows + kernel.rows
                h_total_pad = (base.cols - 1) * stride - base.cols + kernel.cols
                
                v_pad = (v_total_pad // 2, v_total_pad - v_total_pad // 2)
                h_pad = (h_total_pad // 2, h_total_pad - h_total_pad // 2)
                
            case 'full':
                v_pad = (kernel.rows - 1, kernel.rows - 1)
                h_pad = (kernel.cols - 1, kernel.cols - 1)
        
        rows = (base.rows + sum(v_pad) - kernel.rows) // stride + 1
        cols = (base.cols + sum(h_pad) - kernel.cols) // stride + 1
        
        matrix = Matrix(rows, cols)
        
        base = base.expand(v_pad, h_pad)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                ii = i * stride
                jj = j * stride
                
                base_part = [row[jj:jj + kernel.cols] for row in base.data[ii:ii + kernel.rows]]
                
                matrix.data[i][j] = (Matrix.load(base_part) * kernel).sum()
        
        return matrix
    
    @staticmethod
    def convolve(
        base: 'Matrix', 
        kernel: 'Matrix', 
        stride: int = 1, 
        mode: Literal['valid', 'same', 'full'] = 'valid'
    ) -> 'Matrix':
        ''' Convolve the matrix with a kernel '''
        
        return Matrix.correlate(base, kernel.Rot180, stride, mode)
    
    @property
    def T(self) -> 'Matrix':
        ''' Return the transpose of the matrix '''
        
        matrix = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[j][i] = self.data[i][j]
                
        return matrix
    
    @property
    def Rot180(self) -> 'Matrix':
        ''' Rotate the matrix by 180 degrees '''
        
        matrix = Matrix(self.rows, self.cols)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[i][j] = self.data[self.rows - i - 1][self.cols - j - 1]
        
        return matrix
    
    def expand(self, vert: tuple[int, int], horiz: tuple[int, int]) -> 'Matrix':
        ''' Expand the matrix by adding rows and columns '''
        
        matrix = Matrix(self.rows + sum(vert), self.cols + sum(horiz))
        
        for i in range(vert[0], self.rows + vert[0]):
            for j in range(horiz[0], self.cols + horiz[0]):
                matrix.data[i][j] = self.data[i - vert[0]][j - horiz[0]]
            
        return matrix

    def compress(self, vert: tuple[int, int], horiz: tuple[int, int]) -> 'Matrix':
        ''' Compress the matrix by removing rows and columns '''
        
        matrix = Matrix(self.rows - sum(vert), self.cols - sum(horiz))
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = self.data[i + vert[0]][j + horiz[0]]

        return matrix
    
    def __str__(self) -> str:
        ''' Return a string representation of the matrix '''
        
        return str(self.data).replace('],', '],\n')