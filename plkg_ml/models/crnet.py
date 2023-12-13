import torch.nn as nn
import torch

#with rssi approach

class cr_net(nn.Module):
    def __init__(self):
        super(cr_net,self).__init__()
        self.rssi_conv1 = nn.Sequential(nn.Conv2d(1,51,kernel_size=(2,1)),
                                   nn.LayerNorm([51,1,1]),
                                   nn.ReLU())

        self.csi_conv1 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(3,3),padding=(0,1)),
                                   nn.LayerNorm([4,1,51]),
                                   nn.ReLU())#[4,1,51]
        self.csi_conv2 = nn.Sequential(nn.Conv2d(4,1,kernel_size=(1,1)),
                                   nn.LayerNorm([1,1,51]),
                                   nn.ReLU())#[1,1,51]
        self.csi_fnn = nn.Sequential(nn.Linear(51,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,51),
                                 nn.LayerNorm(51),
                                 nn.Sigmoid())
        
    def forward(self,x):
        rssi = torch.unsqueeze(x[:,:,:,0],2)
        rssi = torch.permute(rssi,(0,1,3,2))
        csi = x[:,:,:,1:]
        rssi_out = self.rssi_conv1(rssi)
        rssi_out = torch.permute(rssi_out,(0,3,2,1))
        modify_ = torch.cat((csi,rssi_out),2)
        out1 = self.csi_conv1(modify_)
        out2 = self.csi_conv2(out1)
        out2 = torch.squeeze(out2)
        out = self.csi_fnn(out2)

        return out

if __name__ == "__main__":
    x = torch.rand(1,1,2,52)
    test = cr_net()
    hold = test(x)
    print(hold.size())

