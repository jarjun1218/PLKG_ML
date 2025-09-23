from relation import *
from torch import nn
# from training_core import *
from training_core_copy import *
import torch
from torch.utils.data import Dataset,DataLoader

# normalized training set
trainingset = np.load("normalized_training_set_2.npy") #訓練資料集
# trainingset_0db = np.load("test_set_4/normalized_speed_training_set_0db.npy") #訓練資料集
# trainingset_10db = np.load("test_set_4/normalized_speed_training_set_10db.npy") #訓練資料集
# trainingset_20db = np.load("test_set_4/normalized_speed_training_set_20db.npy") #訓練資料集
# trainingset_30db = np.load("test_set_4/normalized_speed_training_set_30db.npy") #訓練資料集
# trainingset_40db = np.load("test_set_4/normalized_speed_training_set_40db.npy") #訓練資料集
# trainingset_50db = np.load("test_set_4/normalized_speed_training_set_50db.npy") #訓練資料集
# trainingset_all = np.load("test_set_4/normalized_speed_training_set_all.npy") #訓練資料集
# validationset = np.load("test_set_4/normalized_speed_validation_set.npy") #驗證資料集

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

# training with different Noise level
# name = ["", "_0db", "_10db", "_20db", "_30db", "_40db", "_50db", "_all"]
# s = [trainingset, trainingset_0db, trainingset_10db, trainingset_20db, trainingset_30db, trainingset_40db, trainingset_50db, trainingset_all]
# for i in range(len(s)):
#     for pair in model_data_dict:
#         d_s = pair[1](s[i])
#         # v_s = pair[1](validationset)
#         d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#         # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
#         v_l = None
#         # training(pair[0], d_l, v_l, pair[2]+name[i], epoch=200, learning_rate=0.001, quantization=False)

#     for pair in model_data_cnn_dict:
#         d_s = pair[1](s[i])
#         # v_s = pair[1](validationset)
#         d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#         # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
#         v_l = None
#         # training(pair[0], d_l, v_l, pair[2]+name[i], epoch=200, learning_rate=0.001, quantization=False)

# quantized training set
trainingset = np.load("quantified_training_set_2.npy") #訓練資料集
# # validationset = np.load("test_set_4/quantified_speed_validation_set.npy") #驗證資料集
# trainingset_0db = np.load("test_set_4/quantified_speed_training_set_0db.npy") #訓練資料集
# trainingset_5db = np.load("test_set_4/quantified_speed_training_set_5db.npy") #訓練資料集
# trainingset_10db = np.load("test_set_4/quantified_speed_training_set_10db.npy") #訓練資料集
# trainingset_15db = np.load("test_set_4/quantified_speed_training_set_15db.npy") #訓練資料集
# trainingset_20db = np.load("test_set_4/quantified_speed_training_set_20db.npy") #訓練資料集
# trainingset_25db = np.load("test_set_4/quantified_speed_training_set_25db.npy") #訓練資料集
# trainingset_30db = np.load("test_set_4/quantified_speed_training_set_30db.npy") #訓練資料集
# trainingset_35db = np.load("test_set_4/quantified_speed_training_set_35db.npy") #訓練資料集
# trainingset_40db = np.load("test_set_4/quantified_speed_training_set_40db.npy") #訓練資料集
# trainingset_45db = np.load("test_set_4/quantified_speed_training_set_45db.npy") #訓練資料集
# trainingset_50db = np.load("test_set_4/quantified_speed_training_set_50db.npy") #訓練資料集
# trainingset_all = np.load("test_set_4/quantified_speed_training_set_all.npy") #訓練資料集

# s = [trainingset, trainingset_0db, trainingset_10db, trainingset_20db, trainingset_30db, trainingset_40db, trainingset_50db, trainingset_all]

for pair in model_data_cnn_quan_dict:
    d_s = pair[1](trainingset)
    # v_s = pair[1](validationset)
    d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=False, shuffle=True)
    # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True)
    v_l = None
    # training(pair[0], d_l, v_l, pair[2], epoch=200, learning_rate=0.001, quantization=True)

# for i in range(len(s[:1])):
#     for pair in model_data_quan_dict:
#         d_s = pair[1](s[i])
#         # v_s = pair[1](validationset)
#         d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#         # v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True) 
#         v_l = None
#         # training(pair[0], d_l, v_l, pair[2]+name[i], epoch=500, learning_rate=0.001, quantization=True)

#     for pair in model_data_cnn_quan_dict:
#         d_s = pair[1](s[i])
#         # v_s = pair[1](validationset)
#         d_l = DataLoader(dataset=d_s, batch_size=128, pin_memory=True, shuffle=True)
#         # v_l = DataLoader(dataset=v_s, batch_size=128, pin_memory=True, shuffle=True) 
#         v_l = None
#         # training(pair[0], d_l, v_l, pair[2]+name[i], epoch=200, learning_rate=0.005, quantization=True)

# quantized hover training set
# trainingset = np.load("test_set_4/quantified_speed_hover_training_set.npy") #訓練資料集
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
# trainingset = np.load("test_set_4/quantified_speed_moving_training_set.npy") #訓練資料集
# validationset = np.load("test_set_4/quantified_speed_moving_validation_set.npy") #驗證資料集

# for pair in model_data_quan_moving_dict:
#     d_s = pair[1](trainingset)
#     v_s = pair[1](validationset)
#     d_l = DataLoader(dataset=d_s, batch_size=64, pin_memory=True, shuffle=True)
#     v_l = DataLoader(dataset=v_s, batch_size=64, pin_memory=True, shuffle=True)
#     training(pair[0], d_l, v_l, pair[2], epoch=500, learning_rate=0.001)
