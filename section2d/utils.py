# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
  Description :     Tools for data-loading, early-stopping
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

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.functional import F

        
#################################################################################
def default_loader(path):
    return Image.open(path).convert('L')
    
class MyDataset(Dataset): 
    def __init__(self, labels_path, label_cols, IM_PATH,IM_FORMAT,transform=None, 
                 target_transform=None, scaler=None, loader=default_loader):
        super(MyDataset,self).__init__()
        self.imlabels = pd.read_csv(labels_path,index_col=0,header=0,usecols=label_cols)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader 
        self.label = self.imlabels.values
        
        self.scaler = scaler
        if self.scaler is not None:
            if hasattr(scaler,"data_max_"):
                self.label = self.scaler.transform(self.label)
            else:
                self.label = self.scaler.fit_transform(self.label)
        
        self.IM_PATH = IM_PATH
        self.IM_FORMAT = IM_FORMAT

    def __getitem__(self, index):
        tmp_id = self.imlabels.index[index]
        im_path = os.path.join(self.IM_PATH,'%s.%s'%(tmp_id,self.IM_FORMAT))
        img = self.loader(im_path) 
        if self.transform is not None:
            img = self.transform(img) 
        if self.target_transform is not None:
            label = self.target_transform(self.label[index])
        return img, label

    def __len__(self): 
        return len(self.imlabels)

#################################################################################
# Early stopping

class EarlyStopping:
    def __init__(self, patience=1, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, whole=False):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model,whole)
            self.counter = 0

    def save_checkpoint(self, model, whole=False):
        if whole:
            torch.save(model, self.path)
        else:
            torch.save(model.state_dict(), self.path)

####################################################################################################

class WSELoss(nn.Module):
    """
    Weighted square error loss
    """
    def __init__(self, weights):
        """
        Args:
            weights ([list, ndarray, tensor]): [Weights for multi-output loss]
        """        
        super(WSELoss, self).__init__()
        self.register_buffer('weights', torch.tensor(weights).unsqueeze(1))

    def forward(self, y, y_pred):
        y = F.mse_loss(y, y_pred, reduction='none')
        return torch.mm(y, self.weights).mean()
