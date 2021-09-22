import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BinClassifier(nn.Module):
    def __init__(self):
        super(BinClassifier, self).__init__()
        self.linear = nn.Linear(768, 2) 
        
        
    def forward(self, X):
        X = self.linear(X)
        return X
    
class BinClassifierWithFeatures2(nn.Module):
    def __init__(self):
        super(BinClassifierWithFeatures, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(793, 400),
          nn.BatchNorm1d(400),  
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(400, 2),
        )
        
    def forward(self, X):
        X = self.model(X)
        return X
    

class BinClassifierWithFeatures(nn.Module):
    def __init__(self):
        super(BinClassifierWithFeatures, self).__init__()
        
        self.linear = nn.Linear(793, 2) 
        
    def forward(self, X):
        X = self.linear(X)
        return X