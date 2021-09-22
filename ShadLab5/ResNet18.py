import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        
              
    def forward(self, X):
        identity = X.clone()
        
        X = self.block(X)

        X += identity
        X = self.relu(X)

        return X


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.lower_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
        )
        self.upper_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
         
        self.relu = nn.ReLU()   
        
              
    def forward(self, X):
        identity = X.clone()
        identity = self.upper_block(X)
        
        X = self.lower_block(X)
        X += identity
        X = self.relu(X)

        return X
    
    

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2, padding=1),
        )
        
        self.layer1 = nn.Sequential(
             IdentityBlock(in_channels=64, out_channels=64),
             IdentityBlock(in_channels=64, out_channels=64),
        )
        
        
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, num_id_blocks=1)
        
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, num_id_blocks=1)
        
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, num_id_blocks=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes) 
        
        
    
    def make_layer(self, in_channels, out_channels, num_id_blocks):
        self.layer = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            IdentityBlock(out_channels, out_channels),
        )
        return self.layer
                 
       
    def forward(self, X):
        X = self.head(X)
        
        X = self.layer1(X)
        
        X = self.layer2(X)
        
        X = self.layer3(X)
        
        X = self.layer4(X)
        
        X = self.avg_pool(X)
        X = X.view(X.size(0), -1) 
        X = self.fc(X)
        
        return X
                
       