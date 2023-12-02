import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from torch import Tensor

def dataset_maker(file_path:str,exclude:list,low_f:float,high_f:float,minute):
    bdf_file = file_path
    raw = mne.io.read_raw_bdf(bdf_file,preload=True)
    channels_to_eliminate = raw.ch_names[-2:] # Last two channels are trash. Should be discarded.
    raw = raw.copy().pick_types(eeg=True, exclude=channels_to_eliminate)
    raw.rename_channels({'Fp1-A1': 'Fp1',
                        'Fp2-A2': 'Fp2',
                        'F3-A1':'F3',
                        'F4-A2':'F4',
                        'C3-A1':'C3',
                        'C4-A2': 'C4',
                        'P3-A1': 'P3',
                        'P4-A2': 'P4',
                        'O1-A1': 'O1',
                        'O2-A2':'O2',
                        'F7-A1':'F7',
                        'F8-A2':'F8',
                        'T3-A1':'T3',
                        'T4-A2':'T4',
                        'T5-A1':'T5',
                        'T6-A2':'T6'})
    raw.set_montage("standard_1020")
    raw.filter(l_freq=low_f, h_freq=high_f) 
    
    ica = ICA(n_components=16, random_state=97, max_iter="auto",method='fastica')
    ica.fit(raw)
    #ica.plot_components()
    #raw.plot(duration=30)
    ica.exclude = exclude
    
    ica.apply(raw)
    #raw.plot(duration=30)
    data = raw[::][0] 
    
    data = [single_data[300:] for single_data in data]
    
    data = torch.Tensor(np.array(data))
    new_data = []
    collected_data_count = 0
    for i in range(0, data.shape[1], 600):
   
        if collected_data_count >= minute*10:
            break  
        collected_data = data[:, i:i + 300]
        new_data.append(collected_data)
        collected_data_count += 1
    final_tensor = torch.stack(new_data, dim=0) 
    return final_tensor




def ERP(data,label):

    arr0 = []
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    for a,i in enumerate(data):
        if (label[a] <= 9):
            if (label[a] == 0):
                arr0.append(i)
            if (label[a] == 1):
                arr1.append(i)
            if (label[a] == 2):
                arr2.append(i)
            if (label[a] == 3):
                arr3.append(i)
            if (label[a] == 4):
                arr4.append(i)
            if (label[a] == 5):
                arr5.append(i)
            if (label[a] == 6):
                arr6.append(i)
            if (label[a] == 7):
                arr7.append(i)
            if (label[a] == 8):
                arr8.append(i)
            if (label[a] == 9):
                arr9.append(i)
            
        else:
            raise Exception(f"ERROR! label has a index of {label[a]} should start from 0 to N-1 for N number of classes.")
    
    arr0 = torch.stack(arr0,dim=0)
    arr1 = torch.stack(arr1,dim=0)
    arr2 = torch.stack(arr2,dim=0)
    arr3 = torch.stack(arr3,dim=0)
    arr4 = torch.stack(arr4,dim=0)
    arr5 = torch.stack(arr5,dim=0)
    arr6 = torch.stack(arr6,dim=0)
    arr7 = torch.stack(arr7,dim=0)
    arr8 = torch.stack(arr8,dim=0)
    arr9 = torch.stack(arr9,dim=0)
    
    arr0 = arr0.permute(2,0,1)#(50,300,16) -> (16,50,300) -> 
    arr1=arr1.permute(2,0,1)
    arr2=arr2.permute(2,0,1)
    arr3=arr3.permute(2,0,1)
    arr4=arr4.permute(2,0,1)
    arr5=arr5.permute(2,0,1)
    arr6=arr6.permute(2,0,1)
    arr7=arr7.permute(2,0,1)
    arr8=arr8.permute(2,0,1)
    arr9=arr9.permute(2,0,1)
    
    tupler = torch.stack((arr0,arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9),dim=0)
    arr0 = torch.mean(arr0, dim=1)
    arr1 = torch.mean(arr1, dim=1)
    arr2 = torch.mean(arr2, dim=1)
    arr3 = torch.mean(arr3, dim=1)
    arr4 = torch.mean(arr4, dim=1)
    arr5 = torch.mean(arr5, dim=1)
    arr6 = torch.mean(arr6, dim=1)
    arr7 = torch.mean(arr7, dim=1)
    arr8 = torch.mean(arr8, dim=1)
    arr9 = torch.mean(arr9, dim=1)
    
    final_arr = []
    final_arr.append(arr0)  
    final_arr.append(arr1)
    final_arr.append(arr2)
    final_arr.append(arr3)
    final_arr.append(arr4)
    final_arr.append(arr5)
    final_arr.append(arr6)
    final_arr.append(arr7)
    final_arr.append(arr8)
    final_arr.append(arr9)
    arr = torch.from_numpy(np.array(final_arr))
    return arr,tupler


def label_taker(min_list:list,char_to_num=True):
    returned_list = []
    
    for min in min_list:
        val = _minutes(min)
        returned_list.extend(val)
        
    key = ['이제', '우리', '잘', '하고', '그냥', '생각', '지금', '네', '사람', '진짜']#random.seed(10)
    num_returned = []
    if char_to_num:
        for char in returned_list:
            num_returned.append(key.index(char))
        
        return num_returned
    return returned_list

def _minutes(num:int):
    if num == 15:
        return ['이제', '우리', '잘', '하고', '그냥', '생각', '지금', '네', '사람', '진짜',
              '지금', '우리', '생각', '네', '사람', '하고', '진짜', '이제', '잘', '그냥',
              '그냥', '하고', '우리', '잘', '네', '이제', '사람', '생각', '진짜', '지금',
              '사람', '지금', '그냥', '잘', '하고', '진짜', '이제', '네', '우리', '생각',
              '그냥', '잘', '진짜', '우리', '지금', '네', '하고', '사람', '이제', '생각',
              '그냥', '하고', '잘', '사람', '지금', '생각', '진짜', '우리', '네', '이제',
              '지금', '우리', '그냥', '네', '사람', '이제', '진짜', '잘', '생각', '하고',
              '이제', '진짜', '그냥', '네', '하고', '지금', '생각', '우리', '사람', '잘',
              '사람', '하고', '지금', '우리', '네', '이제', '생각', '진짜', '그냥', '잘',
              '우리', '사람', '네', '그냥', '지금', '이제', '하고', '진짜', '잘', '생각',
              '우리', '하고', '그냥', '잘', '이제', '생각', '네', '진짜', '지금', '사람',
              '진짜', '사람', '하고', '그냥', '지금', '이제', '우리', '잘', '네', '생각',
              '하고', '진짜', '이제', '우리', '생각', '네', '그냥', '잘', '지금', '사람',
              '그냥', '이제', '네', '진짜', '하고', '우리', '사람', '잘', '생각', '지금',
              '우리', '그냥', '이제', '진짜', '생각', '잘', '하고', '지금', '네', '사람']
    if num == 10:
        return ['이제', '우리', '잘', '하고', '그냥', '생각', '지금', '네', '사람', '진짜',
              '지금', '우리', '생각', '네', '사람', '하고', '진짜', '이제', '잘', '그냥',
              '그냥', '하고', '우리', '잘', '네', '이제', '사람', '생각', '진짜', '지금',
              '사람', '지금', '그냥', '잘', '하고', '진짜', '이제', '네', '우리', '생각',
              '그냥', '잘', '진짜', '우리', '지금', '네', '하고', '사람', '이제', '생각',
              '그냥', '하고', '잘', '사람', '지금', '생각', '진짜', '우리', '네', '이제',
              '지금', '우리', '그냥', '네', '사람', '이제', '진짜', '잘', '생각', '하고',
              '이제', '진짜', '그냥', '네', '하고', '지금', '생각', '우리', '사람', '잘',
              '사람', '하고', '지금', '우리', '네', '이제', '생각', '진짜', '그냥', '잘',
              '우리', '사람', '네', '그냥', '지금', '이제', '하고', '진짜', '잘', '생각',]
    
    if num == 30:
        return ['이제', '우리', '잘', '하고', '그냥', '생각', '지금', '네', '사람', '진짜', '지금', '우리', '생각', '네', '사람', '하고', '진짜', '이제', '잘', '그냥', '그냥', '하고', '우리', '잘', '네', '이제', '사람', '생각', '진짜', '지금', '사람', '지금', '그냥', '잘', '하고', '진짜', '이제', '네', '우리', '생각', '그냥', '잘', '진짜', '우리', '지금', '네', '하고', '사람', '이제', '생각', '그냥', '하고', '잘', '사람', '지금', '생각', '진짜', '우리', '네', '이제', '지금', '우리', '그냥', '네', '사람', '이제', '진짜', '잘', '생각', '하고', '이제', '진짜', '그냥', '네', '하고', '지금', '생각', '우리', '사람', '잘', '사람', '하고', '지금', '우리', '네', '이제', '생각', '진짜', '그냥', '잘', '우리', '사람', '네', '그냥', '지금', '이제', '하고', '진짜', '잘', '생각',
     '우리', '하고', '그냥', '잘', '이제', '생각', '네', '진짜', '지금', '사람', '진짜', '사람', '하고', '그냥', '지금', '이제', '우리', '잘', '네', '생각', '하고', '진짜', '이제', '우리', '생각', '네', '그냥', '잘', '지금', '사람', '그냥', '이제', '네', '진짜', '하고', '우리', '사람', '잘', '생각', '지금', '우리', '그냥', '이제', '진짜', '생각', '잘', '하고', '지금', '네', '사람', '사람', '생각', '이제', '하고', '잘', '우리', '진짜', '그냥', '지금', '네', '이제', '지금', '진짜', '하고', '잘', '네', '그냥', '생각', '우리', '사람', '이제', '잘', '지금', '하고', '우리', '네', '진짜', '사람', '그냥', '생각', '진짜', '하고', '우리', '이제', '사람', '잘', '네', '지금', '생각', '그냥', '사람', '하고', '잘', '이제', '우리', '생각', '네', '진짜', '지금', '그냥',
     '생각', '우리', '지금', '이제', '진짜', '잘', '네', '그냥', '사람', '하고', '사람', '진짜', '그냥', '잘', '생각', '우리', '지금', '이제', '네', '하고', '그냥', '사람', '하고', '잘', '네', '우리', '지금', '이제', '생각', '진짜', '진짜', '하고', '잘', '이제', '그냥', '지금', '생각', '사람', '우리', '네', '지금', '잘', '사람', '그냥', '진짜', '우리', '네', '하고', '이제', '생각', '지금', '생각', '진짜', '그냥', '사람', '잘', '이제', '네', '하고', '우리', '지금', '생각', '잘', '우리', '진짜', '이제', '그냥', '사람', '네', '하고', '그냥', '생각', '이제', '잘', '지금', '우리', '진짜', '네', '하고', '사람', '잘', '네', '하고', '그냥', '사람', '우리', '지금', '생각', '진짜', '이제', '그냥', '생각', '잘', '진짜', '이제', '하고', '사람', '지금', '네', '우리',]
        
    
    print("ERROR on minutes!")
    raise Exception("ERROR!")


class dataset(nn.Module):
    def __init__(self,data,label):
        self.data = data
        self.label = label
        
    def __len__(self):
        if len(self.label) != self.data.shape[0]:
            raise Exception(f"data.shape: {self.data.shape} and label.len: {len(self.label)}, and error occured")
        
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        return self.data[idx], self.label[idx]
    
    
    
def trainloop(model,optimizer,loss_fn,dLoader,EPOCH:int,modelLocation:str):
    model.train()
    losses = []
    for epoch in range(EPOCH):
        for data,target in dLoader:
            optimizer.zero_grad()

            output = model(data)
            
            loss = loss_fn(output,target.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"Loss for {epoch + 1} epoch: {loss.item()}")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
    torch.save(model, modelLocation)



        
def shape_check(x:Tensor):
    print(f"Shape of X: {x.shape}")
    return 0