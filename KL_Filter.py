import numpy as np
from numpy.linalg import inv

class KL_Filter():
    def __init__(self, measurements):
        self.measurements = measurements
        self.prev_time = 0
        self.X = np.array([[0],[0], [0], [0]])
        self.ground_truth = np.zeros([4, 1])
        self.rmse = np.zeros([4, 1])
        self.P = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1000, 0],
                [0, 0, 0, 1000]
                ])
        self.A = np.array([
                [1.0, 0, 1.0, 0],
                [0, 1.0, 0, 1.0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0]
                ])
        self.H = np.array([
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0]
                ])
        self.I = np.identity(4)
        self.z = np.zeros([2, 1])
        self.R = np.array([
                [0.0225, 0],
                [0, 0.0225]
                ])
        self.noise_ax = 5
        self.noise_ay = 5
        self.Q = np.zeros([4, 4])
    
    def predict():
        self.X = np.matmul(self.A, self.X)
        At = np.transpose(self.A)
        self.P = np.add(np.matmul(self.A, np.matmul(self.P, At)), self.Q)
        
    def update():    
        # Measurement update step
        Y = np.subtract(self.z, np.matmul(self.H, self.X))
        Ht = np.transpose(self.H)
        S = np.add(np.matmul(self.H, np.matmul(self.P, Ht)), self.R)
        K = np.matmul(self.P, Ht)
        Si = inv(S)
        K = np.matmul(K, Si)

        # New state
        self.X = np.add(self.X, np.matmul(K, Y))
        self.P = np.matmul(np.subtract(self.I, np.matmul(K, self.H)), self.P)
        
    
    
    