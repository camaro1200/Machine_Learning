from itertools import chain, islice
from collections import Counter

import matplotlib.pyplot as plt

import typing as tp
import pandas as pd

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import nltk 
from nltk import tokenize
from nltk.tokenize import WordPunctTokenizer

import re
import string

import gensim.downloader as api
from nltk.tokenize import TweetTokenizer


def load_data(train_path, test_path):
    
    data_train = pd.read_csv(train_path, index_col=None)

    data_test = pd.read_csv(test_path, index_col=None)
    
    data_text_columns = ["keyword", "location", 'text']

    data_train[data_text_columns] = data_train[data_text_columns].fillna('NaN') # cast missing values to string "NaN"

    data_test[data_text_columns] = data_test[data_text_columns].fillna('NaN') # cast missing values to string "NaN"
    
    return data_train, data_test


glove_tokenizer = TweetTokenizer()
glove = api.load("glove-twitter-25")  # load glove vectors
preprocess = lambda text: ' '.join(glove_tokenizer.tokenize(text.lower()))

def vectorize_sum(comment):
    """
    implement a function that converts preprocessed comment to a sum of token vectors
    """
    glove_dim = glove.vector_size
    features = np.zeros([glove_dim], dtype='float32')
    
    comment = preprocess(comment)
    comment = comment.split(' ')
    
    for token in comment:
        if token in glove.key_to_index:
            token = glove.get_vector(token)
            features += token
          
    return features


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.bert_tokenizer = tokenizer
        self.data = data

       
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        text_sample = self.bert_tokenizer(self.data['text'][index])
        #text_sample['text'] = self.data['text'][index]
        
        if 'target' in self.data:
            target = self.data['target'][index]
        else:
            target = None
        
        keyword_sample = torch.tensor(vectorize_sum(self.data['keyword'][index]))    # not smart!!
        
        id_sample = self.data['id'][index]
        location_sample = self.data['location'][index]
        
        return {'input_ids': torch.tensor(text_sample['input_ids']),
                'token_type_ids': torch.tensor(text_sample['token_type_ids']),
                'attention_mask': torch.tensor(text_sample['attention_mask']),
                'keyword': keyword_sample, 
                'location': location_sample,
                'id': id_sample, 
                'target': target }


def get_train_dataset(data, tokenizer):
    text_dataset = []
    for index, row in data.iterrows():
        elem = tokenizer(row['text'])
        elem['input_ids'] = torch.as_tensor(elem['input_ids'])
        elem['token_type_ids'] = torch.tensor(elem['token_type_ids'])
        elem['attention_mask'] = torch.tensor(elem['attention_mask'])
        elem['tokens'] = row['text']
        elem['target'] = row['target']
        elem['id'] = row['id']
        text_dataset.append(elem)

    X_train, X_eval = train_test_split(text_dataset, test_size=0.33, random_state=42)
    
    return X_train, X_eval 



def get_test_dataset(data_test, tokenizer, mask=True):
    test_dataset = []
    for index, row in data_test.iterrows():
        #print(index, row['keyword'], row['location'], row['text'])
        elem = tokenizer(row['text'])
        elem['input_ids'] = torch.as_tensor(elem['input_ids'])
        elem['token_type_ids'] = torch.tensor(elem['token_type_ids'])
        elem['attention_mask'] = torch.tensor(elem['attention_mask'])
        elem['tokens'] = row['text']
        elem['id'] = row['id']
        test_dataset.append(elem)
        
    return test_dataset


class PadSequence:
    def __init__(self, padded_columns, device='cuda:1'):
        self.padded_columns = set(padded_columns)
        self.device = device

    def __call__(self, batch):
        padded_batch = defaultdict(list)
        for example in batch:
            for key, tensor in example.items():
                padded_batch[key].append(tensor)
            
        #print("done")       
        for key, val in padded_batch.items():
            #print(val)
            if key in self.padded_columns:
                padded_batch[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True).to(self.device)
        return padded_batch
    

def clean(title):
    title = re.sub(r"\-"," ",title)
    title = re.sub(r"\+"," ",title)
    title = re.sub (r"&","and",title)
    title = re.sub(r"\|"," ",title)
    title = re.sub(r"\\"," ",title)
    title = re.sub(r"\W"," ",title)
    title = title.lower()
    for p in string.punctuation :
        title = re.sub(r"f{p}"," ",title)
    
    title = re.sub(r"\s+"," ",title)
    
    return title
    
 


