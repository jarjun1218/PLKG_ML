import numpy as np
import torch
from torch.utils.data import Dataset
#pure CSI
class csi_dataset(Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:]).float()
        self.length = len(self.uavdata)
    def __getitem__(self, index):
        data, label = self.uavdata[index], self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length



#csi with rssi approach
class csi_rssi_dataset(Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:]).float()
        self.length = len(self.uavdata)
    def __getitem__(self, index):
        data, label = self.uavdata[index], self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length

#csi cnn approach
class csi_cnn_dataset(Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:]).float()
        self.length = len(self.uavdata)-1
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    

# our approach
#STA,AP,STA
class csi_rssi_cnn_dataset(Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:]).float()
        self.length = len(self.uavdata)-1
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    

if __name__ == "__main__":
    test1 = csi_cnn_dataset(np.load('normalized_training_set.npy'))
    test2 = csi_rssi_cnn_dataset(np.load('normalized_training_set.npy'))
    print("csi_cnn_dataset:",test1.__getitem__(0)[0].size())
    print("csi_cnn_rssi_dataset:",test2.__getitem__(0)[0].size())