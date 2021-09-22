import numpy as np
import os
import matplotlib.pyplot as plt 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import random
from torch.utils.data import DataLoader

def create_dataset():
    os.chdir('/home/iris/paulshab/ImageClassification/')
    t = Compose([
        #Resize((150, 150)),
        Resize((227, 227)),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    
    train_dataset = ImageFolder("archive/seg_train/", transform=t)
    val_dataset = ImageFolder("archive/seg_test/", transform=t)
    
    return train_dataset, val_dataset
    
    
def create_dataloader(train_dataset, val_dataset):
    trainloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    testloader = DataLoader(
        val_dataset,
        batch_size=16 ,
        shuffle=True,
        num_workers=8,
        drop_last=True
     )
    return trainloader, testloader


