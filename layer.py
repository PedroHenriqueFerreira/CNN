from abc import ABC, abstractmethod

class Layer:
    ''' Abstract class for layers '''
    
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input): 
        ''' Propagates input forward '''
        
        raise NotImplementedError()
    
    def backward(self, gradient):
        ''' Propagates gradient backward '''
        
        raise NotImplementedError()