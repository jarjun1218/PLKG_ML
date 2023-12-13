from relation import *
from torch import nn
from training_core import *
import torch
from torch.utils.data import Dataset,DataLoader

trainingset = np.load("normalized_training_set.npy")

for pair in model_data_cnn_dict:
    d_s = pair[1](trainingset)
    d_l  = DataLoader(dataset=d_s,batch_size=16,pin_memory=True,shuffle=True)
    training(pair[0],d_l,pair[2])


for pair in model_data_dict:
    d_s = pair[1](trainingset)
    d_l  = DataLoader(dataset=d_s,batch_size=16,pin_memory=True,shuffle=True)
    training(pair[0],d_l,pair[2])