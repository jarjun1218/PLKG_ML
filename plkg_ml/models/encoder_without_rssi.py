import torch.nn as nn
import torch

#with rssi
class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(51,32),
            nn.LayerNorm(32),
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
        output = self.model(x)
        return output
