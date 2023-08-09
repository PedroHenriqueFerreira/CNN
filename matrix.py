from typing import Callable, Union
from random import uniform

class Matrix:
    ''' Matrix class for matrix operations '''
    
    def __init__(self, rows: int, cols: int):
        ''' Initialize a matrix with the given dimensions '''
        
        if rows <= 0 or cols <= 0:
            raise ValueError('Matrix dimensions must be positive non-zero integers')
        
        self.rows = rows
        self.cols = cols
        
        self.data = [[0.0] * cols for _ in range(rows)]
    
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
        
    def __str__(self) -> str:
        ''' Return a string representation of the matrix '''
        
        return str(self.data).replace('],', '],\n')
    
    @property
    def T(self) -> 'Matrix':
        ''' Return the transpose of the matrix '''
        
        matrix = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.data[j][i] = self.data[i][j]
                
        return matrix