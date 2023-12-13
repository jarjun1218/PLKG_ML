import torch.nn as nn
import torch

class cnn_basic(nn.Module):
    def __init__(self):
        super(cnn_basic,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([4,1,51]),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(4,1,kernel_size=(1,1)),
                                   nn.LayerNorm([1,1,51]),
                                   nn.ReLU())
        self.fnn = nn.Sequential(nn.Linear(51,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,51),
                                 nn.LayerNorm(51),
                                 nn.Sigmoid())
        
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out2 = torch.squeeze(out2)
        out = self.fnn(out2)

        return out

if __name__ == "__main__":
    x = torch.rand(1,2,51)
    test = cnn_basic()
    hold = test(x)
    print(hold.size())

