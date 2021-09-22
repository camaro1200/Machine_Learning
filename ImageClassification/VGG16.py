import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, pretrained=False, state_dict_path=None):
        super(VGG16, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # ( in_channels, out_channels, window_size, )
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            
        )
        self.fc_model = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 6)
        )
        
        self.pretrained = pretrained
        if self.pretrained:
            self.load_state_dict(torch.load(state_dict_path))    

    def forward(self, X):
        # X.shape == (batch_size, 512, 7, 7)
        X = self.cnn_model(X)
        X = X.view(X.size(0), -1) 
        X = self.fc_model(X)  
        return X
        