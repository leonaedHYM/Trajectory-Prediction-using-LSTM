from __future__ import print_function, division
import torch
from torch import nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import time
import math
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
from tqdm import tqdm
from utils import ngsimDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class Maneuver_class(nn.Module):
    
    def __init__(self, input_size, hidden_size=128, num_layers = 1):
        
        super(Maneuver_class, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, 64)
        self.hidden_size = 128
        self.num_layers = 1

        # define LSTM layer
        self.lstm = nn.LSTM(64, hidden_size = 128,            # input 是 (sequence_size, batch_size, input_size)
                            num_layers = 1, batch_first=False)       
        self.lat_linear = nn.Linear(self.hidden_size, 3) 
        self.lon_linear = nn.Linear(self.hidden_size, 2) 
        
        #define activation:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x_input):
        
        embedded = self.embedding(x_input)
        embedded = self.leaky_relu(embedded)
        lstm_out, self.hidden = self.lstm(embedded)
        lat_temp = self.lat_linear(lstm_out[-1])
        lat_pred = self.softmax(lat_temp)
        
        lon_temp = self.lon_linear(lstm_out[-1])
        lon_pred = self.softmax(lon_temp)
        
        
        return  lat_pred, lon_pred
    
    
def valid(model, valid_loader, loss_fn):
    model.eval()
    valid_loss = 0.
    for data in tqdm(valid_loader):
        hist, mask, lat_enc, lon_enc, fut, op_mas = data
        hist,nbrs,lat_enc, lon_enc= hist.to(device), nbrs.to(device), lat_enc.to(device), lon_enc.to(device)
        lat_pred, lon_pred = model(hist)    
        lat_loss = loss_fn(lat_pred, lat_enc)
        lon_loss = loss_fn(lon_pred, lon_enc)
            
        loss = lat_loss + lon_loss
        valid_loss += loss.item()
    return valid_loss / len(valid_loader)


def train_model(data_dir, input_size, n_epochs=100, _batch_size=1, _lr = 0.001, load_model=False, model_path='output/model0.pth'):
    
    trSet = ngsimDataset(data_dir+'TrainSet.mat')
    valSet = ngsimDataset(data_dir+'ValSet.mat')
    trDataloader = DataLoader(trSet,batch_size=_batch_size,shuffle=True,collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet,batch_size=_batch_size,shuffle=True,collate_fn=valSet.collate_fn)
    # training
    model = Maneuver_class(input_size).to(device) # 目前尝试只考虑本车：input_size=2

    if load_model == True:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
        
    min_loss = 100.  # why?

    for epoch in range(n_epochs):
        total_loss = 0.
        it = 0
        print(f"Epoch {epoch+1}\n-------------------------------")
 
        for data in tqdm(trDataloader):
            hist, mask, lat_enc, lon_enc, fut, op_mas = data
            hist,lat_enc, lon_enc = hist.to(device), lat_enc.to(device), lon_enc.to(device)  #缺一个padding方法,目前input_size要设置为2
            lat_pred, lon_pred = model(hist)
            
            lat_loss = loss_fn(lat_pred, lat_enc)
            lon_loss = loss_fn(lon_pred, lon_enc)
            
            loss = lat_loss + lon_loss
            # zero the gradient
            optimizer.zero_grad()
                          
            # backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            it += 1
            if it%500==0:
                print("Training Iteration {} of epoch {} complete. Loss: {}".
                    format(it, epoch, loss.item()))

        epoch_loss = total_loss/len(trDataloader)
        print('epoch:{},avg_loss:{}'.format(epoch, epoch_loss))
        #save best model
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                           'output/model_{}.pth'.format(epoch))
            print("epoch:%d Model Saved" % epoch)

        #validation
        if epoch % 10 == 0:
            valid_loss = valid(model, valDataloader, loss_fn)
            print('validation epoch:{},valid_loss:{}'.format(epoch, valid_loss))
            model.train()
            
data_dir = '/mnt/e/Northwestern University/Courses/2022winter/pattern recognition/projects/data/'
train_model(data_dir, 2, n_epochs=100, _batch_size=1, _lr = 0.001, load_model=False, model_path='output/model0.pth')