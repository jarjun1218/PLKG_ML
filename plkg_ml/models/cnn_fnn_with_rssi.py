from torch import nn
import torch


class cnn_fnn_rssi(nn.Module):
    def __init__(self):
        super(cnn_fnn_rssi,self).__init__()
        self.fnn = nn.Sequential(
            nn.Conv2d(1,1,(2,1),stride=1,padding=0),
            nn.LayerNorm(52),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128,51),
            nn.LayerNorm(51),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
    
    #weight initialize
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    def forward(self, x):
        output = self.fnn(x)
        return output
