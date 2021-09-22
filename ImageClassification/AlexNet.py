import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, pretrained=False, state_dict_path=None):
        super(AlexNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2),                          # kernel size, stride
            nn.Conv2d(96, 256, kernel_size = 5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2), 
            nn.Conv2d(256, 384, kernel_size = 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size = 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size = 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2),
        )
        self.fc_model = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 6)
        )
        
        self.pretrained = pretrained
        if self.pretrained:
            self.load_state_dict(torch.load(state_dict_path))
        

    def forward(self, X):
        # X.shape == (batch_size, 3, 227, 227)
        X = self.cnn_model(X)
        X = X.view(X.size(0), -1) 
        X = self.fc_model(X)  
        return X


