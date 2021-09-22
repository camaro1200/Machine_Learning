import numpy as np
import matplotlib.pyplot as plt 
import random
from torch.nn.functional import sigmoid
import torch
import torch.nn as nn
from train import parse_yml
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle  

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street' ]
m = nn.Softmax(dim=1)

def display_images(dataset):
    k = random.randint(0, len(dataset))
    img = np.array(dataset[k][0])
    img = np.transpose(img, (1, 2, 0))
    plt.title(class_names[dataset[k][1]])
    plt.axis("off")
    plt.imshow(img);

    
def check_forward_pass(model, data_loader):
    _, _, device = parse_yml()
    net_batch = model.to(device)
    
    batch = next(iter(data_loader))
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)
    
    logits = net_batch(images)
    return logits


def convergence_graph(loss_vals):
    print("min loss value", loss_vals[-1])

    plt.plot(range(1, len(loss_vals) +1), loss_vals, color ='blue')
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("cost (J)")
    plt.title("Convergence of gradient descent")


def check_cuda(model, device):
    if next(model.parameters()).is_cuda == False:
        model.to(device)
        

def check_accuracy(model, dataloader):
    
    _, _, device = parse_yml()
    check_cuda(model, device)
    model.eval()
    
    mistake_img = []
    mistake_label = []
    true_label = []
    pred = []
    lbl = []
    correct = 0
    total = 0
    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            y_pred = model(images)
        
        y_pred = m(y_pred)
        _, y_pred = y_pred.max(1)
        
        correct += (y_pred == labels).sum()
        total += y_pred.size(0)
        
        mistake = (y_pred == labels)
 
        for i in range(mistake.shape[0]):
            if mistake[i] == False:
                mistake_img.append(images[i].cpu().numpy())
                mistake_label.append(y_pred[i].cpu().numpy())
                true_label.append(labels[i].cpu().numpy())
        
        pred.append(y_pred.cpu().numpy())
        lbl.append(labels.cpu().numpy())
        

    print('Accuracy of the network on the images: %d %%' % (100 * correct / total))
    return mistake_img, mistake_label, true_label, pred, lbl 
    


def display_false_images(img_set, label_set, true_set):
    k = random.randint(0, len(img_set))
    img = img_set[k]
    img = np.transpose(img, (1, 2, 0))
    print("predicted class:", class_names[label_set[k]])
    print("true class:", class_names[true_set[k]])
    plt.rcParams["axes.grid"] = False
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
    
def confusion_matrix_ut(lbl, pred):
        
    CM = np.zeros((6, 6))
    N = len(pred)
    for i in range(N):
        CM += confusion_matrix(lbl[i], pred[i], labels=[0, 1, 2, 3, 4, 5])
    ax = plt.axes()
    sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
    ax.set_title('Confusion matrix')
    plt.show()
    

