from math import log

from .matrix import Matrix

class Loss:
    ''' Loss function '''
    
    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        ''' Return the loss of the matrix '''
        
        raise NotImplementedError()
    
    def gradient(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        ''' Return the gradient of the matrix '''
        
        raise NotImplementedError()


class MeanSquaredLoss(Loss):
    ''' Squared loss function '''
    
    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        return (0.5 * (y_true - y_pred) ** 2).mean()

    def gradient(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        return y_true - y_pred

class CrossEntropyLoss(Loss):
    ''' Cross entropy loss function '''

    def loss(self, y_true: Matrix, y_pred: Matrix) -> float:
        y_pred = y_pred.map(lambda x: max(1e-15, min(1 - 1e-15, x)))
        
        y_pred_log = y_pred.map(lambda x: log(x))
        y_pred_neg_log = y_pred.map(lambda x: log(1 - x))
        
        return - (y_true * y_pred_log + (1 - y_true) * y_pred_neg_log).mean()

    def gradient(self, y_true: Matrix, y_pred: Matrix) -> Matrix:
        y_pred = y_pred.map(lambda x: max(1e-15, min(1 - 1e-15, x)))
        
        return y_true / y_pred - (1 - y_true) / (1 - y_pred)
