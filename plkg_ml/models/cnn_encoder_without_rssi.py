from torch import nn
import torch


class cnn_encoder(nn.Module):
    def __init__(self):
        super(cnn_encoder,self).__init__()
        self.fnn = nn.Sequential(
            nn.Conv2d(1,1,(2,1),stride=1,padding=0),
            nn.LayerNorm(51),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32,51),
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
