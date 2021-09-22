from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

criterion = nn.CrossEntropyLoss()      # has softmax inside

def parse_yml():
    yaml_file = open("parameters.yaml")
    parsed_yaml_file = yaml.load(yaml_file)
    dic = parsed_yaml_file["param_dic"]
    
    dev_num = str(dic['device'])
    device = torch.device(("cuda:" + dev_num) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    return dic['epochs'], dic['lr'], device
    

def train_model(net, dataloader, save_weights=False, state_dict_path=None):
    num_epochs, lr, device = parse_yml()
    net.to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    loss_arr = []
    for epoch in tqdm(range(num_epochs)):
        loss_val = 0
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
        
            #forward
            y_pred = net(images)
            loss = criterion(y_pred,labels)
            loss_val = loss + loss_val
        
            #backward
            optimizer.zero_grad()
            loss.backward()
        
            #gradient decent
            optimizer.step()
        
        loss_val = loss_val/16
        loss_arr.append(loss_val)
        
    if save_weights:
        torch.save(net.state_dict(), state_dict_path)
        
    return loss_arr


def sanity_check_train(model, dataloader):
    num_epochs, lr, device = parse_yml()
    
    model = model.to(device)
    model.train()
    
    batch_optimizer = optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(dataloader))
    images, labels = batch
    images, labels = images[:1].to(device), labels[:1].to(device)

    loss_vals = []
    for epoch in tqdm(range(num_epochs)):
        y_pred = model(images.float().requires_grad_(True))
    
        loss = criterion(y_pred, labels)

        batch_optimizer.zero_grad()
        loss.backward()
        batch_optimizer.step()

        loss_vals.append(loss.item())
    return loss_vals

