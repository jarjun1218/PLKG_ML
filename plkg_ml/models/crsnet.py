import torch.nn as nn
import torch

#with rssi approach

class crs_net(nn.Module):
    def __init__(self):
        super(crs_net,self).__init__()

        self.rssi_conv1 = nn.Sequential(nn.Conv2d(1,51,kernel_size=(2,1)),
                                   nn.LayerNorm([51,1,1]),
                                   nn.ReLU())
        
        self.rssi_fnn = nn.Sequential(nn.Linear(1, 16),
                                      nn.LayerNorm(16),
                                      nn.ReLU(),
                                      nn.Linear(16, 51),
                                      nn.LayerNorm(51),
                                      nn.ReLU())

        self.speed_conv1 = nn.Sequential(nn.Conv2d(1,51,kernel_size=(2,1)),
                                      nn.LayerNorm([51,1,1]),
                                      nn.ReLU())
        
        self.speed_fnn = nn.Sequential(nn.Linear(1, 16),
                                       nn.LayerNorm(16),
                                       nn.ReLU(),
                                       nn.Linear(16, 51),
                                       nn.LayerNorm(51),
                                       nn.ReLU())
        
        self.csi_conv1 = nn.Sequential(nn.Conv2d(1,1,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([1,1,51]),
                                   nn.ReLU())#[1,1,51]
        
        self.csi_conv2 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([4,1,51]), #changed from [4,1,51]
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
        # rssi = torch.unsqueeze(x[:,:,:,0],2)
        # rssi = torch.permute(rssi,(0,1,3,2))
        # rssi_out = self.rssi_conv1(rssi)
        # rssi_out = torch.permute(rssi_out,(0,3,2,1))

        # speed = torch.unsqueeze(x[:,:,:,52],2)
        # speed = torch.permute(speed,(0,1,3,2))
        # speed_out = self.speed_conv1(speed)
        # speed_out = torch.permute(speed_out,(0,3,2,1))

        rssi = x[:,:,:,0]
        print(rssi.size())
        rssi = torch.unsqueeze(x[:,:,:,0],2)
        print(rssi.size())
        rssi = torch.permute(rssi,(0,1,3,2))
        print(rssi.size())
        rssi_out = self.rssi_fnn(rssi)
        print(rssi_out.size())
        rssi_out = torch.permute(rssi_out,(0,3,2,1))
        print(rssi_out.size())

        speed = torch.unsqueeze(x[:,:,:,52],2)
        speed = torch.permute(speed,(0,1,3,2))
        speed_out = self.speed_fnn(speed)
        speed_out = torch.permute(speed_out,(0,3,2,1))
        # print(speed_out.size())

        csi = x[:,:,:,1:52]
        csi_ = self.csi_conv1(csi)
        # print(csi_.size())

        modify = torch.cat((csi_,speed_out),dim=2)
        # modify_ = torch.cat((modify,rssi_out),dim=2)
        out2 = self.csi_conv2(modify)
        out3 = self.csi_conv3(out2)
        out4 = torch.squeeze(out3)
        out = self.csi_fnn(out4)

        return out

if __name__ == "__main__":
    x = torch.rand(1,1,2,53)
    test = crs_net()
    hold = test(x)
    print(hold.size())

