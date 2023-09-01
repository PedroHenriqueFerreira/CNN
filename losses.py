from matrix import Matrix

class Loss:
    ''' Loss function '''
    
    def __call__(self, y: Matrix, y_pred: Matrix) -> Matrix:
        ''' Return the loss of the matrix '''
        
        raise NotImplementedError()
    
    def gradient(self, y: Matrix, y_pred: Matrix) -> Matrix:
        ''' Return the gradient of the matrix '''
        
        raise NotImplementedError()

class MeanSquaredLoss(Loss):
    ''' Squared loss function '''
    
    def __call__(self, y: Matrix, y_pred: Matrix) -> Matrix:
        return 0.5 * (y - y_pred) ** 2

    def gradient(self, y: Matrix, y_pred: Matrix) -> Matrix:
        return y_pred - y

class CrossEntropyLoss(Loss):
    ''' Cross entropy loss function '''

    def __call__(self, y: Matrix, y_pred: Matrix) -> Matrix:
        y_pred = y_pred.clip(1e-15, 1 - 1e-15)
        
        return - y * y_pred.log() - (1 - y) * (1 - y_pred).log()

    def gradient(self, y: Matrix, y_pred: Matrix) -> Matrix:
        y_pred = y_pred.clip(1e-15, 1 - 1e-15)
        
        return - (y / y_pred) + (1 - y) / (1 - y_pred)
