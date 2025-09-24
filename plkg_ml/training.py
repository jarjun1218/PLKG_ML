from relation import *
from torch import nn
# from training_core import *
from training_core_copy import *
import torch
from torch.utils.data import Dataset,DataLoader

# normalized training set
trainingset = np.load("normalized_training_set_2.npy") #訓練資料集

for pair in model_data_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=32, pin_memory=True, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
    v_l = None
    # training(pair[0], d_l, v_l, pair[2], epoch=200)

for pair in model_data_cnn_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=False, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True)
    v_l = None
    training(pair[0], d_l, v_l, pair[2], epoch=200, learning_rate=0.001, quantization=False)

# quantized training set
trainingset = np.load("quantified_training_set_2.npy") #訓練資料集

for pair in model_data_cnn_quan_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=False, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True)
    v_l = None
    # training(pair[0], d_l, v_l, pair[2], epoch=200, learning_rate=0.001, quantization=True)


