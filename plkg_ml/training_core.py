#training core
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np
import os

def training(model,data_loader,folder,checkpoint = 100):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    epoch = 500
    Model = model()
    Model.cuda()
    Model.float()
    optimizer = optim.Adam(Model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    MSE_loss = nn.MSELoss(reduction="mean")
    MSE_loss.cuda()
    Model.train()
    n_iters = []
    losses = []
    count = 0
    for n_iter in tqdm(range(epoch), desc = "training"):
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            output = Model(data)

            loss = MSE_loss(output, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        
        n_iters.append(n_iter)
        losses.append(train_loss)
        count += 1
        if count % checkpoint == 0:
            file_name = folder + "/model"+str(count)+".pth"
            torch.save(Model,file_name)
    torch.save(Model,folder + "/model_final.pth")

    np.save(folder +"/losses",np.array([n_iters,losses]))
        

def testing(model, data_loader):
    Model = model
    Model = Model.cuda()
    Model.eval()
    total_loss = 0
    MSE_loss = nn.MSELoss(reduction="mean")
    for data,target in data_loader:
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = MSE_loss(output, target)
        total_loss += loss.item()

    return total_loss

def original(data_loader):
    total_loss = 0
    MSE_loss = nn.MSELoss(reduction="mean")
    for data,target in data_loader:
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        loss = MSE_loss(data, target)
        total_loss += loss.item()

    return total_loss
