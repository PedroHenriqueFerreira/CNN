from scipy import signal

print(signal.correlate([1, 2, 3], [0, 1, 0.5], mode='same', method=''))