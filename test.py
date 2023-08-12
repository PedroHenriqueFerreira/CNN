from scipy import signal

signal.correlate2d([[1,2,3],[4,5,6],[7,8,9]], [[1,2],[3,4]], mode='same')