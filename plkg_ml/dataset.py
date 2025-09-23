import numpy as np
from numpy.core.multiarray import array as array
import torch
from torch.utils.data import Dataset

class base_Dataset(Dataset):
    def __init__(self, data: np.array):
        self.uavrssi = torch.from_numpy(data[0,:,1]).float()
        self.iotrssi = torch.from_numpy(data[1,:,1]).float()
    def __getRssi__(self, index):
        data, label = self.uavrssi[index], self.iotrssi[index]
        return data, label


#pure CSI
class csi_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:52]).float()  #uav|iot,data_count,rssi+csi
        self.iotdata = torch.from_numpy(data[1,:,1:52]).float()
        self.length = len(self.uavdata)
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index], self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length

#csi cnn approach
class csi_cnn_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:52]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:52]).float()
        self.length = len(self.uavdata)-1
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length     
    
# csi cnn with lstm
class csi_cnn_lstm_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,:]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:52]).float()
        self.length = len(self.uavdata)-1
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    
# quantized dataset
# basic csi
class csi_quan_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:103]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:103]).float()
        self.length = len(self.uavdata)-1
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index], self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    
#csi cnn quantization approach
class csi_cnn_quan_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:103]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:103]).float()
        self.length = len(self.uavdata)-1
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    
#csi cnn speed quantization approach
class csi_cnn_speed_quan_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,1:104]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:103]).float()
        self.length = len(self.uavdata)-1
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+2].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length
    
# quantized dataset with LSTM
class csi_cnn_quan_lstm_dataset(base_Dataset):
    def __init__(self, data: np.array):
        self.uavdata = torch.from_numpy(data[0,:,:]).float()
        self.iotdata = torch.from_numpy(data[1,:,1:103]).float()
        self.length = len(self.uavdata)-2
        super().__init__(data)
    def __getitem__(self, index):
        data, label = self.uavdata[index:index+3].unsqueeze(0), self.iotdata[index]
        return data, label
    def __len__(self):
        return self.length

    

if __name__ == "__main__":
    test1 = csi_cnn_dataset(np.load('normalized_training_set.npy'))
    # test2 = csi_rssi_cnn_dataset(np.load('normalized_training_set.npy'))
    print("csi_cnn_dataset:",test1.__getitem__(0)[0].size())
    # print("csi_cnn_rssi_dataset:",test2.__getitem__(0)[0].size())