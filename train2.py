import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from torch.utils.data import Dataset,DataLoader
from utils_neuro import *
from models import *
import librosa
from tqdm import trange

test5 = dataset_maker('data/test5_15.BDF',[0,1,11,13],0.01,40,15)
test6 = dataset_maker('data/test6.BDF',[0,1,5,9],0.01,40,10)
test7 = dataset_maker('data/test7.BDF',[1,2,3,4,5,8,11],0.01,40,10)
test8 = dataset_maker('data/test8_30.BDF',[1,3,7,8,12],0.01,40,30)
#test9 = dataset_maker('data/test9_5.BDF',[0],0.01,40,5) #-> it's under 5 min. must be error.
test10 = dataset_maker('data/test10_15.BDF',[1,3,4,8,9,12],0.01,40,15)
test11 = dataset_maker('data/test11_15.BDF',[2,4,5],0.01,40,15)
test12 = dataset_maker('data/test12.BDF',[0,1,4,6,9],0.01,40,10)
test13 = dataset_maker('data/test13_15.BDF',[1,4,5,11,13],0.01,40,15)
test14 = dataset_maker('data/test14_15.BDF',[0,1,2,4,5,6],0.01,40,15)
test15 = dataset_maker('data/test15_15.BDF',[0,1,3,8,12],0.01,40,15)
test16 = dataset_maker('data/test16_15.BDF',[0,1,2,3,4,5,7,12,13],0.01,40,15)
test17 = dataset_maker('data/test17_15.BDF',[0,1,2,3,9],0.01,40,15)

final_data = torch.cat((test5,test6,test7,test8,test10,test11,test12,test13,test14,test15,test16,test17),dim=0)
final_data = final_data.permute(0,2,1)

print(f"final_data.shape: {final_data.shape}") #torch.Size([1800, 300, 16])
labels = label_taker([15,10,10,30,15,15,10,15,15,15,15,15])
print(f"len(labels) : {len(labels)}")

#dataloader
mean = torch.mean(final_data, dim=(0, 1, 2), keepdim=True)
std = torch.std(final_data, dim=(0, 1, 2), keepdim=True)
normalized_data = (final_data - mean) / std + 3*6.3152e-06
normalized_data = normalized_data * 10
dat = dataset(normalized_data[:-100], labels[:-100])
dLoader = DataLoader(dat,batch_size=32,shuffle=True)
print(normalized_data.shape)

class lstmattn(nn.Module):
    '''
        input: (batch,seq,channels)
        output: (batch, 10)
        lstm
        gelu
        lstm
        fc1
        dropout
        fc2
    '''
    def __init__(self):
          super().__init__()

          self.layer1 = nn.LSTM(16,10,num_layers=1,batch_first=True)
          self.gelu = nn.GELU()
          self.layer2 = nn.LSTM(10,4,num_layers=1,batch_first=True)
          self.fc1 = nn.Sequential(
               nn.Linear(300*4, 600),
               nn.GELU(),
               nn.Dropout(),
               nn.Linear(300*2,300),
               nn.GELU(),
               nn.Dropout(),
               nn.Linear(300,128),
               nn.GELU(),
               nn.Linear(128,64),
               nn.GELU(),
               nn.Linear(64,10)
          )
     
    def forward(self, x):
        x,_ = (self.layer1(x))
        x = self.gelu(x)
        x,_ = self.layer2(x)
        x = self.gelu(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
#param

EPOCH = 100000
model = lstmattn()
optimizer = optim.Adam(model.parameters(),lr=3e-4,betas=(0.9,0.999))
loss_fn = nn.CrossEntropyLoss()

import time


trainloop(model=model,
          optimizer=optimizer,
          loss_fn=loss_fn,
          dLoader=dLoader,
          EPOCH=EPOCH,
          modelLocation="new_model.pth")
new_model = torch.load("new_model.pth")



dat2 = dataset(normalized_data[-100:], labels[-100:])
dLoader2 = DataLoader(dat2,batch_size=32,shuffle=False,drop_last=True)

stack = 0
k = 0
new_model.eval()
with torch.no_grad():
    for data,target in dLoader2:
        output = new_model(data)
        max_indices = torch.argmax(output, dim=1)
        print(max_indices)
        print(target)

        for i in range(32):
            k += 1
            if max_indices[i] == target[i]:
                stack += 1
                
    print(f"total correction: {(stack/k)*100}%")
    