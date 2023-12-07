import torch.nn as nn
import torch

#without rssi
class fnn(nn.Module):
    def __init__(self):
        super(fnn,self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(51,128),
            nn.LayerNorm(128),
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
