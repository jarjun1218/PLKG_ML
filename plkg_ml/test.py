from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import *


testset = np.load("plkg_ml/normalized_testing_set.npy")
model = torch.load('plkg_ml/fnn/model_final.pth')
model.to("cpu")
model.eval()
d_s = csi_dataset(testset)
data, iot_original = d_s.__getitem__(100)
index = torch.from_numpy(np.array([i+1 for i in range(51)]))
uav_original = data
uav_modify = model(data).detach().numpy()
plt.plot(index,iot_original,label = "iot_o")
plt.plot(index,uav_original,label = "uav_o")
plt.plot(index,uav_modify,label = "uav_m")
plt.legend(title = "name")
plt.show()
