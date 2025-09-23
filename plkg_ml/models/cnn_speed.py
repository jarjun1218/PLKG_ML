import torch.nn as nn
import torch

#with rssi approach

class cs_net(nn.Module):
    def __init__(self):
        super(cs_net,self).__init__()
        
        self.speed_fnn = nn.Sequential(nn.Linear(1,51),
                                      nn.LayerNorm(51),
                                      nn.ReLU())

        self.speed_conv1 = nn.Sequential(nn.Conv2d(1,51,kernel_size=(2,1)),
                                      nn.LayerNorm([51,1,1]),
                                      nn.ReLU())
        
        self.csi_conv1 = nn.Sequential(nn.Conv2d(1,1,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([1,1,51]),
                                   nn.ReLU())#[1,1,51]
        
        self.csi_conv2 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([4,1,51]),
                                   nn.ReLU())#[4,1,51]
        
        self.csi_conv3 = nn.Sequential(nn.Conv2d(4,1,kernel_size=(1,1)),
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
    
    def forward(self, x):
        speed = torch.unsqueeze(x[:,:,:,51],2)
        speed = torch.permute(speed,(0,1,3,2))
        speed_out = self.speed_conv1(speed)
        speed_out = torch.permute(speed_out,(0,3,2,1))

        csi = x[:,:,:,:51]
        csi_ = self.csi_conv1(csi)
        # print(csi_.size())
        
        modify_ = torch.cat((csi_,speed_out),2)
        # print(modify_.size())
        out2 = self.csi_conv2(modify_)
        # print(out2.size())
        out3 = self.csi_conv3(out2)
        # print(out3.size())
        out4 = torch.squeeze(out3)
        # print(out4.size())
        out = self.csi_fnn(out4)
        # print(out.size())

        return out

if __name__ == "__main__":
    x = torch.rand(1,1,2,52)
    test = cs_net()
    hold = test(x)
    print(hold.size())

