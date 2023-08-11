from typing import Callable, Union, Literal, Any
from random import uniform

import numpy as np

class Matrix:
    ''' Matrix class for matrix operations '''
    
    def __init__(self, *shape: int):
        ''' Initialize a matrix with the given dimensions '''
        
        if any(dim < 1 for dim in shape):
            raise ValueError('Invalid shape dimensions')
        
        self.shape = shape
        
        self.data = self.create_data(shape)
    
    def create_data(self, shape: tuple[int, ...]) -> list[Any]:
        if len(shape) == 0:
            return 0 # type: ignore
        
        return [self.create_data(shape[1:]) for _ in range(shape[0])]
            
    def __str__(self) -> str:
        ''' Return a string representation of the matrix '''
        
        return str(self.data).replace('],', '],\n')

print(Matrix(4, 3, 2))
print()
# print(np.zeros((4, 3, 2)))