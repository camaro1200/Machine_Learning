import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import statistics
import wandb
import os
import typing as tp


def convergence_graph(loss_vals):
    print("min loss value", loss_vals[-1])

    plt.plot(range(1, len(loss_vals) +1 ), loss_vals, color ='blue')
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("cost (J)")
    plt.title("Convergence of gradient descent")
    
    
def get_sentence_pooler_output(model, batch):
    model.eval()
    with torch.no_grad():
        return model(input_ids=batch["input_ids"],
                          token_type_ids=batch["token_type_ids"],
                          attention_mask=batch["attention_mask"],
                          return_dict=True).pooler_output


def get_sentence_embeddings(model, batch):
    model.eval()
    with torch.no_grad():
        return model(input_ids=batch["input_ids"],
                          token_type_ids=batch["token_type_ids"],
                          attention_mask=batch["attention_mask"],
                          return_dict=True)["last_hidden_state"][:,0]
    

def apply_mask(example: tp.Dict[str, tp.Any], device='cuda:0'):
    
    # copy real input_ids:
    example['labels'] = example['input_ids'].detach().clone()
    
    # create random array of floats in equal dimension to input_ids
    rand = torch.rand(example['input_ids'].shape).to(device)

    # where the random array is less than 0.15, we set true
    mask_arr = rand < 0.2 *  (example['input_ids'] != 101) * (example['input_ids'] != 102) * (example['input_ids'] != 0) 
    
    # create selection from mask_arr
    selection = mask_arr.nonzero().tolist()

    for i, j in selection:
        example['input_ids'][i][j] = 103
        
    return example
    
    
def train_bin_class(classifier, model, dataloader, num_epochs=10, device='cuda:1', lr=0.001, pooler=False):
    classifier.to(device)
    classifier.train()
    
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    train_loss = []
    
    #wandb.init(project="kaggle_nlp", name="binary_classifier")
    
    for epoch in tqdm(range(num_epochs)):
        avg_batch_loss = []
        for batch in dataloader: 
            labels = torch.tensor(batch['target']).to(device)

            with torch.no_grad():
                if pooler == True:
                    cls_logits = model(input_ids=batch["input_ids"],
                          token_type_ids=batch["token_type_ids"],
                          attention_mask=batch["attention_mask"]).pooler_output
                else:   
                    cls_logits = model(input_ids=batch["input_ids"],
                          token_type_ids=batch["token_type_ids"],
                          attention_mask=batch["attention_mask"],
                          return_dict=True)["last_hidden_state"][:,0]
            
            logits = torch.cat((cls_logits, batch['keyword']), dim=1)
            logits = classifier(logits)
            
            loss = criterion(logits, torch.squeeze(labels.T))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            avg_batch_loss.append(loss.item())
            
        avg = sum(avg_batch_loss) / len(avg_batch_loss)
        #loss_dict = {'train_loss': loss.item()}
        #wandb.log(loss_dict, step=epoch)
        train_loss.append(avg)
    
    #wandb.finish()
    return train_loss

def pretrain_masked_lm(model, loader, num_epochs=10, device='cuda:1'):
    model.to(device)
    model.train() 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_loss = []
    avg_batch_loss = []
    for epoch in tqdm(range(num_epochs)):
        avg_batch_loss = []
        for batch in loader:
            batch = apply_mask(batch)
            outputs = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels'].long(), 
                        output_hidden_states=True,
                        return_dict=True
                       )
            loss = outputs.loss
            #print(loss)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_batch_loss.append(loss.item())
        
        avg = sum(avg_batch_loss) / len(avg_batch_loss)
        train_loss.append(avg)
        
    return train_loss


def finetune_model(model, trainloader, num_epochs=10, device='cuda:1'):
    model.to(device)
    model.train() 
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loss = []

    for epoch in tqdm(range(num_epochs)):
        avg_batch_loss = []
        for batch in trainloader: 
            logits = model(batch['input_ids'], batch['attention_mask']).logits[:,0].to(device)
            #print(logits.shape)
            target = torch.tensor(batch['target']).to(device)
            
            loss = criterion(logits, target.long())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            avg_batch_loss.append(loss.item())
            
        avg = sum(avg_batch_loss) / len(avg_batch_loss)
        #print(avg)
        train_loss.append(avg)
        
    return train_loss


def eval_model(model, dataloader, classifier=None, device='cuda:1', generate_csv=False, mode=0):
    
    if classifier != None:
        classifier.to(device)
        classifier.eval()
    
    model.to(device)
    model.eval()
    
    df = pd.DataFrame(columns=['id', 'target'])
    
    scores = {"accuracy": [], "f1_score": [], "recall": [], "precision": []}
    for batch in tqdm(dataloader):
        with torch.no_grad():
            if mode == 0:
                cls_logits = get_sentence_embeddings(model, batch)
                logits = torch.cat((cls_logits, batch['keyword']), dim=1)
                logits = classifier(logits) 
            elif mode == 1:
                cls_logits = get_sentence_pooler_output(model, batch)
                logits = torch.cat((cls_logits, batch['keyword']), dim=1)
                logits = classifier(logits) 
            elif mode == -1:
                logits = model(batch['input_ids'], batch['attention_mask']).logits[:,0]
                #print(logits.shape)
                                           
            #print(cls_logits)
            prob = F.softmax(logits, -1)                                    # get log-SoftMax
            prob = torch.argmax(prob, -1)                                   # get argMax
            prob = prob.detach().cpu().tolist()                             # to cpu
            #print(prob)
        
        labels = batch['target']
        scores["accuracy"].append(accuracy_score(labels, prob))
        scores["f1_score"].append(f1_score(labels, prob, average='macro'))
        scores["recall"].append(recall_score(labels, prob))
        scores["precision"].append(precision_score(labels, prob)) 
            
 
    return {
            "accuracy":  statistics.mean(scores['accuracy']), 
            "f1_score": statistics.mean(scores['f1_score']), 
            "recall": statistics.mean(scores['recall']),
            "precision": statistics.mean(scores['precision']),
            }


            