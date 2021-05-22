# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-------------------------------------------------
  Description :     Backbone Net for Training
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

##################################################################################################

class FullConnectLayer(nn.Module):
    def __init__(self,layer_elems,actfunc=nn.ReLU(),bnorm=False,dropout=None):
        super(FullConnectLayer, self).__init__()
        fc = []
        pre_elems = layer_elems.pop(0)
    
        for j in layer_elems:
            fc.append(nn.Linear(pre_elems, j))
            fc.append(actfunc)
            
            if bnorm:
                fc.append(nn.BatchNorm1d(j))
            if dropout is not None:
                fc.append(dropout)
            
            pre_elems = j
        
        self.fc = nn.Sequential(*fc)
        
    def forward(self,x):
        return self.fc(x)

##################################################################################################

class ResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.LeakyReLU(negative_slope=0.05,inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,inchannels=1,layers=[1,3,4,2],unit_channels=16,num_class=1000):
        super(ResNet,self).__init__()
        self._inchannels = inchannels
        self._unit_channels = unit_channels

        self.pre = nn.Sequential(
            nn.Conv2d(self._inchannels,self._unit_channels,3,1,1,bias=True),
            nn.BatchNorm2d(self._unit_channels),
            nn.LeakyReLU(negative_slope=0.05,inplace=True),

            nn.Conv2d(self._unit_channels,self._unit_channels,3,2,1,bias=False),
            nn.BatchNorm2d(self._unit_channels),
            nn.LeakyReLU(negative_slope=0.05,inplace=True),

            nn.Conv2d(self._unit_channels,self._unit_channels,3,1,1,bias=True),
            nn.BatchNorm2d(self._unit_channels),
            nn.LeakyReLU(negative_slope=0.05,inplace=True),

            nn.AvgPool2d(3,2,1)
        )

        self.layer1 = self._make_layer(self._unit_channels,self._unit_channels,layers[0]) 
        self.layer2 = self._make_layer(self._unit_channels*4,self._unit_channels*2,layers[1],2) 
        self.layer3 = self._make_layer(self._unit_channels*8,self._unit_channels*4,layers[2],2)
        self.layer4 = self._make_layer(self._unit_channels*16,self._unit_channels*8,layers[3],2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(self._unit_channels*32,num_class),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,inchannels,unit_channels,blocks,stride=1):
        downsample = None
        if stride != 1 or inchannels != unit_channels * ResidualBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inchannels, unit_channels * ResidualBlock.expansion, kernel_size=1, stride=1),
                nn.BatchNorm2d(unit_channels * ResidualBlock.expansion),
                nn.AvgPool2d(3,stride,1),
            )

        layers = []
        layers.append(ResidualBlock(inchannels,unit_channels,stride,downsample))
        outchannels = unit_channels * ResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResidualBlock(outchannels,unit_channels))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0),-1)
        return self.fc(x)

