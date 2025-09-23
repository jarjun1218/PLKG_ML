import torch.nn as nn
import torch

#with rssi approach

class cs_net(nn.Module):
    def __init__(self):
        super(cs_net,self).__init__()
        
        self.speed_emb = nn.Embedding(num_embeddings=3,embedding_dim=51)

        # self.speed_transform = nn.Sequential(nn.Linear(51,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,51),nn.ReLU())
        
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
        
        self.alpha = 0.8
    
    def forward(self, x):
        speed_labels = x[:,:,1:,51].long()
        speed_emb = self.speed_emb(speed_labels)
        # speed_emb = self.speed_transform(speed_emb)
        # speed_emb = (1 - self.alpha) * speed_emb

        csi = x[:,:,:,:51]
        csi_ = self.csi_conv1(csi)
        # csi_ = self.alpha * csi_
        
        modify_ = torch.cat((csi_,speed_emb),2)
        out2 = self.csi_conv2(modify_)
        out3 = self.csi_conv3(out2)
        out4 = torch.squeeze(out3)
        out = self.csi_fnn(out4)

        return out

if __name__ == "__main__":
    x = torch.rand(1,1,2,52)
    test = cs_net()
    hold = test(x)
    print(hold.size())

