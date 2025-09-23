import torch.nn as nn
import torch

#without rssi
class fnn(nn.Module):
    def __init__(self):
        super(fnn,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(102,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256,102),
            nn.LayerNorm(102),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
    
    #weight initialize
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    def forward(self, x):
        output = self.model(x)
        return output
