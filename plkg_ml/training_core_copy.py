#training core
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np
import os
import greycode_quantization as quan

def training(model, data_loader, val_loader ,folder, checkpoint=100, epoch=500, learning_rate=0.001):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    epoch = epoch
    lr = learning_rate
    Model = model()
    Model.cuda()
    Model.float()
    optimizer = optim.Adam(Model.parameters(), lr, betas=(0.9, 0.999), eps=1e-8) #optimizer
    BCE_loss = nn.BCELoss(reduction="mean")
    BCE_loss.cuda()
    MSE_loss = nn.MSELoss(reduction="mean")
    MSE_loss.cuda()
    # state_dict = Model.state_dict()
    # for key, param in state_dict.items():
    #     with torch.no_grad():
    #         param.fill_(0)
    # Model.load_state_dict(state_dict)
    Model.train()
    n_iters = []
    losses = []
    val_losses = []
    best_val_loss = 100000
    count = 0
    for n_iter in tqdm(range(epoch), desc = "training"):
        # training phase
        train_loss = 0.0
        batch_idx_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target) #
            data, target = data.cuda(), target.cuda() #

            optimizer.zero_grad() #

            output = Model(data)

            # loss = MSE_loss(output, target) #loss function
            loss = BCE_loss(output, target)

            loss.backward() #gradient descent

            optimizer.step() #to next step

            train_loss += loss.item() #loss value
            batch_idx_count +=1

        # validation phase
        Model.eval()
        val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = Variable(data), Variable(target)
                    data, target = data.cuda(), target.cuda()
                    output = Model(data)
                    loss = BCE_loss(output, target)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        n_iters.append(n_iter)
        losses.append(train_loss/batch_idx_count)
        count += 1
        if count % checkpoint == 0:
            file_name = folder + "/model"+str(count)+".pth"
            torch.save(Model,file_name)
    torch.save(Model,folder + "/model_final.pth")

    np.save(folder +"/losses",np.array([n_iters,losses]))
    if val_loader is not None:
        np.save(folder +"/val_losses",np.array([n_iters,val_losses]))
        

def testing(model, data_loader):
    Model = model
    Model = Model.cuda()
    total_loss = 0
    MSE_loss = nn.MSELoss(reduction="mean")
    batch_idx_count = 0
    for data,target in data_loader:
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = MSE_loss(output, target)
        total_loss += loss.item()
        batch_idx_count += 1

    return total_loss/batch_idx_count
