import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from utils_neuro import *
import librosa
from tqdm import trange
from models import *

import numpy as np
from mne import create_info, concatenate_raws
from mne.io import RawArray

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
tensor_data = final_data.permute(0,2,1)
print(f"final_data.shape: {final_data.shape}") #torch.Size([1800, 300, 16])
labels = label_taker([15,10,10,30,15,15,10,15,15,15,15,15])
print(f"len(labels) : {len(labels)}") # (1800,)

a,tupler = ERP(final_data, labels)
print(len(tupler))
print(tupler.shape)
print(f"tupler.shape: {tupler.shape}")
data = tupler[0,:,:,:]
# Channel names and mapping
channel_mapping = {
    'Fp1-A1': 'Fp1',
    'Fp2-A2': 'Fp2',
    'F3-A1': 'F3',
    'F4-A2': 'F4',
    'C3-A1': 'C3',
    'C4-A2': 'C4',
    'P3-A1': 'P3',
    'P4-A2': 'P4',
    'O1-A1': 'O1',
    'O2-A2': 'O2',
    'F7-A1': 'F7',
    'F8-A2': 'F8',
    'T3-A1': 'T3',
    'T4-A2': 'T4',
    'T5-A1': 'T5',
    'T6-A2': 'T6'
}

# Create info object
ch_names = list(channel_mapping.values())
sfreq = 100  # Replace this with your actual sampling frequency
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Create RawArray
data = data.permute((2,1,0))
data = data.reshape(16,-1)
raw = mne.io.RawArray(data, info)

# Plot 10 words on one image
raw.plot(n_channels=len(ch_names), scalings='auto', title='EEG Data')

# Show the plot
plt.show()
