#training core
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np
import os
import greycode_quantization as quan


def kdr(uav,iot):
    uav = list(uav)
    iot = list(iot)
    count = 0
    for u,i in zip(uav,iot):
        if u != i:
            count += 1
    return count/len(uav)


def KDR(model, data_loader, Nbits, inbits, guard = 0):
    Model = model
    Model = Model.cuda()
    Model.eval()
    batch_idx_count = 0
    O_KDR = 0
    MAP_KDR = 0
    for data,target in data_loader:
        batch_idx_count += 1
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        hold0 = data.cpu().detach().numpy()
        uav_key = quan.quantization_1(hold0,Nbits,inbits,guard)
        output = torch.squeeze(output)
        hold1 = output.cpu().detach().numpy()
        uav_key_map = quan.quantization_1(hold1,Nbits,inbits,guard)
        target = torch.squeeze(target)
        hold2 = target.cpu().detach().numpy()
        iot_key = quan.quantization_1(hold2,Nbits,inbits,guard)
        original_kdr = kdr(uav_key,iot_key)
        map_kdr = kdr(uav_key_map,iot_key)
        O_KDR += original_kdr
        MAP_KDR += map_kdr

    return (O_KDR/batch_idx_count,MAP_KDR/batch_idx_count)


def kdr_rssi(model, data_loader, Nbits, inbits, guard = 0):
    Model = model
    Model = Model.cuda()
    Model.eval()
    batch_idx_count = 0
    O_KDR = 0
    MAP_KDR = 0
    for data,target in data_loader:
        batch_idx_count += 1
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        hold0 = data.cpu().detach().numpy()
        uav_key = quan.quantization_1(hold0,Nbits,inbits,guard)
        output = torch.squeeze(output)
        hold1 = output.cpu().detach().numpy()
        uav_key_map = quan.quantization_1(hold1,Nbits,inbits,guard)
        target = torch.squeeze(target)
        hold2 = target.cpu().detach().numpy()
        iot_key = quan.quantization_1(hold2,Nbits,inbits,guard)
        original_kdr = kdr(uav_key,iot_key)
        map_kdr = kdr(uav_key_map,iot_key)
        O_KDR += original_kdr
        MAP_KDR += map_kdr

    return (O_KDR/batch_idx_count,MAP_KDR/batch_idx_count)




