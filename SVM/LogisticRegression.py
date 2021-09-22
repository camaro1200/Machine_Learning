import numpy as np
from numpy.random import rand
from sklearn.metrics import accuracy_score
import pandas as pd

def add_bias(X):
    X = pd.DataFrame(X)
    X = pd.concat([pd.Series(1, index=X.index, name='00'), X], axis=1)
    X = np.array(X)
    return X


def xavier(in_, out_):
    return np.random.randn(in_, out_) * np.sqrt(2. / (in_ + out_))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(X, y, weights):                 
    z = np.dot(X, weights)        
    predict_1 = y * np.log(sigmoid(z))
    predict_0 = (1 - y) * np.log(1 - sigmoid(z))        
    return -sum(predict_1 + predict_0) / len(X)


class LogisticRegression:
    def __init__(self, X, y, random_init=False):
        self.X = add_bias(X)
        self.y = y.reshape(-1,1)
        self.M = X.shape[0]
        self.N = X.shape[1]
        if random_init == False:
            self.w = xavier(self.N, 1)
        else:
            self.w = np.zeros((self.N, 1))
  
        
    
    def fit(self, epochs=1000, lr=0.05):
        cost_val = []
        
        for i in range(epochs):
            j = cost_function(self.X, self.y, self.w)  
            cost_val.append(j)
            #print(j)
    
            y_hat = sigmoid(np.dot(self.X, self.w))                                   
            self.w -= lr * np.dot(self.X.T,  y_hat - self.y) / self.M                 
    
        return cost_val
    
    def predict(self, X1, y1):
        pred = np.around(sigmoid(X1 @ self.w))
        acc = accuracy_score(y1, pred)
        return acc