from relation import *
from torch import nn
# from training_core import *
from training_core_copy import *
import torch
from torch.utils.data import Dataset,DataLoader

# normalized training set
trainingset = np.load("test_set_4/normalized_speed_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/normalized_speed_validation_set.npy") #驗證資料集

# standardized training set
# trainingset = np.load("test_set_4/standardized_speed_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/standardized_speed_validation_set.npy") #驗證資料集

# for pair in model_data_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
#     training(pair[0], d_l, v_l, pair[2], epoch=100)

# for pair in model_data_cnn_speed_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
#     training(pair[0], d_l, v_l, pair[2], epoch=500)

# for pair in model_data_cnn_speed_dict:
#     d_s = pair[1](trainingset)
#     d_l  = DataLoader(dataset=d_s,batch_size=128,pin_memory=True,shuffle=True)
#     training(pair[0],d_l,pair[2])

# trainingset = np.load("test_set_4/normalized_speed_label_training_set.npy")

# for pair in model_data_cnn_speed_label_dict:
#     d_s = pair[1](trainingset)
#     d_l  = DataLoader(dataset=d_s,batch_size=128,pin_memory=True,shuffle=True)
#     training(pair[0],d_l,pair[2])

# quantized training set shuffled
trainingset = np.load("test_set_4/quantified_speed_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/quantified_speed_validation_set_shuffled.npy") #驗證資料集

for pair in model_data_quan_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=64, pin_memory=True, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True) 
    v_l = None
    training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)

# quantized training set
trainingset = np.load("test_set_4/quantified_speed_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/quantified_speed_validation_set.npy") #驗證資料集

for pair in model_data_cnn_quan_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
    v_l = None
    # training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)

for pair in model_data_cnn_lstm_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True)
    v_l = None
    # training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)

# quantized hover training set
trainingset = np.load("test_set_4/quantified_speed_hover_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/quantified_speed_hover_validation_set.npy") #驗證資料集

# for pair in model_data_quan_hover_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=64, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True)
#     training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)

# for pair in model_data_cnn_quan_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=64, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True)
#     training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)

# # quantized moving training set
trainingset = np.load("test_set_4/quantified_speed_moving_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/quantified_speed_moving_validation_set.npy") #驗證資料集

# for pair in model_data_quan_moving_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=64, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True)
#     training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)
