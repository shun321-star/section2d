# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
  Description :     Main script for the whole work
  Author :          shun.321
  Mail :            shun.321@sjtu.edu.cn
  Date :            2021.5.22
-------------------------------------------------
  Change Activity:
          None
-------------------------------------------------
"""
__author__ = 'shun.321'


import torch
import torchvision
import torch.nn as nn
import numpy as np 
import pandas as pd

import os
import sys
import datetime
import logging
import pickle

from utils import EarlyStopping, WSELoss, MyDataset
from backbone import ResNet

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from functools import partial
from sklearn.preprocessing import MinMaxScaler


#########################################################################################################

IM_PATH = './img'
IM_FORMAT = 'png'
NOW = datetime.datetime.now().strftime('%Y-%m-%d-%H')

logging.basicConfig(
    filename = './Log/'+ NOW +'.log',
    format = '[%(asctime)s:%(message)s]', 
    level = logging.INFO,
    filemode = 'a',
    datefmt = '%H:%M:%S'
)

#############################################################################################################

# train and evaluate
def eval_acc(test_iter,net,metrics=nn.MSELoss(),device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            l = metrics(net(X.to(device)),y.to(device))
            acc_sum += l.cpu().item()
            n += 1
    net.train()
    
    return acc_sum / n

def train(net, train_iter, test_iter, loss, num_epochs = 30,
            optimizer = None, early_stopping = False,
            device = None, sheduler =None,
            checkpoint_dir = 'checkpoint.pt'):
    
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    
    print('training on ',device)
    logging.info("training on {}".format(device))
    logging.info("Structure of the net:\n{}".format(str(net)))

    if optimizer is None:
        optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001)
    
    if early_stopping:
        early_stopping_ = EarlyStopping(patience=100, path=checkpoint_dir)
    
    loss = loss.to(device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            net.train()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            if sheduler is not None:
                sheduler.step(epoch+1)

            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
        net.eval()

        train_loss = train_l_sum/batch_count
        test_loss = eval_acc(test_iter,net,device=device)
        
        if early_stopping:
            early_stopping_((train_loss+test_loss)/2, net, whole=True)
            if early_stopping_.early_stop:
                print('Early Stopping')
                print(f'Best validation loss: {early_stopping_.best_score}')
                break  
       
        print('epoch %d, train_loss %.4f, test_loss %.4f'%(epoch+1, train_loss, test_loss))
        logging.info('epoch %d, train_loss %.4f, test_loss %.4f'%(epoch+1, train_loss, test_loss))
    
    print(f'Best validation loss: {early_stopping_.best_score}')
    logging.info(f'Best validation loss: {early_stopping_.best_score}')
#####################################################################################################

train_im_trans = transforms.Compose([
    transforms.ToTensor(),
])


test_im_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])

target_scaler = MinMaxScaler((0.0,10.0))
target_trans = transforms.Compose([
    partial(torch.tensor,dtype=torch.float32),
])


####################################################################################################
# Task 1: E

train_data = MyDataset('./data/target.csv', [0,1],IM_PATH,IM_FORMAT,transform=train_im_trans, 
                        target_transform=target_trans, scaler=target_scaler)
test_data = MyDataset('./data/target.csv', [0,1],IM_PATH,IM_FORMAT,transform=test_im_trans, 
                        target_transform=target_trans, scaler=train_data.scaler)

with open('./models/YoungModulesTrans.pkl', 'wb') as f:
    pickle.dump(train_data.scaler, f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet(unit_channels=32,num_class=1)

loss = nn.MSELoss()
lr, num_epochs, batch_size = 0.001, 100, 256

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(net.parameters(),lr=lr)
train(net, train_loader, test_loader, loss, num_epochs,
            optimizer = optimizer, early_stopping = True,
            device = device, sheduler =None,
            checkpoint_dir = './models/YoungModules.pth')

###################################################################################################
# Task 2: Strain

target_scaler = MinMaxScaler((0.0,10.0))

train_data = MyDataset('./data/target.csv', [0,2,4,6,8,10],IM_PATH,IM_FORMAT,transform=train_im_trans, 
                        target_transform=target_trans, scaler=target_scaler)
test_data = MyDataset('./data/target.csv', [0,2,4,6,8,10],IM_PATH,IM_FORMAT,transform=test_im_trans, 
                        target_transform=target_trans, scaler=train_data.scaler)

with open('./models/StrainTrans.pkl', 'wb') as f:
    pickle.dump(train_data.scaler, f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet(unit_channels=32,num_class=5)

loss = nn.MSELoss()
lr, num_epochs, batch_size = 0.001, 100, 256

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(net.parameters(),lr=lr)
train(net, train_loader, test_loader, loss, num_epochs,
            optimizer = optimizer, early_stopping = True,
            device = device, sheduler =None,
            checkpoint_dir = './models/Strain.pth')


###################################################################################################
# Task 2: Stress

target_scaler = MinMaxScaler((0.0,10.0))

train_data = MyDataset('./data/target.csv', [0,3,5,7,9,11],IM_PATH,IM_FORMAT,transform=train_im_trans, 
                        target_transform=target_trans, scaler=target_scaler)
test_data = MyDataset('./data/target.csv', [0,3,5,7,9,11],IM_PATH,IM_FORMAT,transform=test_im_trans, 
                        target_transform=target_trans, scaler=train_data.scaler)

with open('./models/StressTrans.pkl', 'wb') as f:
    pickle.dump(train_data.scaler, f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ResNet(unit_channels=32,num_class=5)

loss = nn.MSELoss()
lr, num_epochs, batch_size = 0.001, 100, 256

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(net.parameters(),lr=lr)
train(net, train_loader, test_loader, loss, num_epochs,
            optimizer = optimizer, early_stopping = True,
            device = device, sheduler =None,
            checkpoint_dir = './models/Stress.pth')