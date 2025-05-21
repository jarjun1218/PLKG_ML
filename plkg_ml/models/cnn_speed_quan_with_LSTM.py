import torch.nn as nn
import torch

#with speed and LSTM pridiction approach
class LSTMNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=3, num_layers=2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

class LSTM_model_init():
    def __init__(self, input_size=3, hidden_size=32, output_size=3, num_layers=2):
        self.model = LSTMNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('LSTM_model/model_final.pth'))
        self.model.eval()
    
    def predict(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
        return output
    
class cs_net(nn.Module, LSTM_model_init):
    def __init__(self):
        super(cs_net,self).__init__()
        
        self.pos_fnn = nn.Sequential(nn.Linear(3,32),
                                    nn.LayerNorm(32),
                                    nn.ReLU(),
                                    nn.Linear(32,32),
                                    nn.LayerNorm(32),
                                    nn.ReLU(),
                                    nn.Linear(32,102),
                                    nn.LayerNorm(102),
                                    nn.ReLU())
        
        self.speed_fnn = nn.Sequential(nn.Linear(1,32),
                                        nn.LayerNorm(32),
                                        nn.ReLU(),
                                        nn.Linear(32,102),
                                        nn.LayerNorm(102),
                                        nn.ReLU())

        self.speed_conv1 = nn.Sequential(nn.Conv2d(1,102,kernel_size=(2,1)),
                                      nn.LayerNorm([102,1,1]),
                                      nn.ReLU())
        
        self.csi_conv1 = nn.Sequential(nn.Conv2d(1,1,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([1,1,102]),
                                   nn.ReLU())#[1,1,51]
        
        self.csi_conv2 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(3,5),padding=(0,2)),
                                   nn.LayerNorm([4,1,102]),
                                   nn.ReLU())#[4,1,51]
        
        self.csi_conv3 = nn.Sequential(nn.Conv2d(4,1,kernel_size=(1,1)),
                                   nn.LayerNorm([1,1,102]),
                                   nn.ReLU())#[1,1,51]
        
        self.csi_fnn = nn.Sequential(nn.Linear(108,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,102),
                                 nn.LayerNorm(102),
                                 nn.Sigmoid())
    
    def forward(self, x):
        # speed = torch.unsqueeze(x[:,:,:,103],2)
        # speed = torch.permute(speed,(0,1,3,2))
        # print(speed.size())
        # speed_out = self.speed_conv1(speed)
        # speed_out = torch.permute(speed_out,(0,3,2,1))
        # speed_out = self.speed_fnn(speed)
        # print(speed_out.size())

        pos = x[:,:,1,104:116]
        # print(pos.size())
        pos = torch.unsqueeze(pos,1)
        # print(pos.size())
        pos_ = torch.tensor([]).to(x.device)
        for i in range(4):
            p = torch.tensor([]).to(x.device)
            for j in range(3):
                p = torch.cat((p,pos[:,:,:,i*3+j]),2)
            pos_ = torch.cat((pos_,p),1)
        
        pos_ = pos_.unsqueeze(1)
        # print(pos_.size())
        pos_r = pos_[:,:,3,0:3]
        pos_p = LSTM_model_init().predict(torch.squeeze(pos_, 1)).to(x.device).unsqueeze(1)
        # print(pos_r)
        # print(pos_p)
        # print(pos_p - pos_r)
        pos_m = pos_p - pos_r
        vel = pos_m / 0.1
        vel = torch.squeeze(vel)
        # pos_p = torch.unsqueeze(pos_p,1)
        # pos_r = torch.unsqueeze(pos_r,1)
        # print(pos_p.size()) # [1, 3]
        # print(pos_r.size()) # [1, 3]
        pos_ = torch.cat((pos_,torch.unsqueeze(pos_p,1)),2)
        # print(pos_.size()) # [2, 3]
        pos_out = self.pos_fnn(pos_[:,:,3:,:])
        pos_p = torch.squeeze(pos_p)
        pos_r = torch.squeeze(pos_r)
        # pos_out = self.pos_conv(torch.permute(pos_,(0,2,3,1)))
        # print(pos_out.size())
        

        csi = x[:,:,:,1:103]
        # csi = torch.cat((csi,pos_),3)
        # print(csi.size())
        # csi_ = self.csi_conv1(csi)
        # print(csi_.size())
        
        # modify_ = torch.cat((csi_,speed_out,pos_out),2)
        # print(modify_.size())
        out2 = self.csi_conv2(csi)
        # print(out2.size())
        out3 = self.csi_conv3(out2)
        # print(out3.size())
        out4 = torch.squeeze(out3)
        out4 = torch.cat((out4, pos_p, vel),-1)
        # print(out4.size())
        out = self.csi_fnn(out4)
        # print(out.size())

        return out
    

    
if __name__ == "__main__":
    x = torch.rand(32,1,3,116)
    # LSTM_model_init()
    test = cs_net()
    hold = test(x)
    print(hold.size())

