from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import math
#from padding_batching import padding_batching
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = True

writer = SummaryWriter('runs/batch_128')

# Initialize network
net = highwayNet(args, 14)
#net = nn.DataParallel(net, device_ids=[0,1,2,3])
if args['use_cuda']:
    net = net.cuda()


## Initialize optimizer
pretrainEpochs = 500
trainEpochs = 10000
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#optimizer = nn.DataParallel(optimizer, device_ids=[0,1,2,3])
batch_size = 128
crossEnt = torch.nn.BCELoss()


## Initialize data loaders
trSet = ngsimDataset('data/TrainSet.mat')
valSet = ngsimDataset('data/ValSet.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=valSet.collate_fn)
#valDataloader = None
#val_datadir = 'data/ValSet.mat'
#tr_datadir = 'data/trSet.mat'
#val_set = padding_batching(val_datadir, batch_size)
#tr_set = padding_batching(tr_datadir, batch_size)

#tr_set = torch.load('valset_batchSize=64.pt')
print('-----------training starts-------------')

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
#prev_val_loss = math.inf
min_loss = 100.0

for epoch in range(pretrainEpochs+trainEpochs):

    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True
    st = time.time()
    # Variables to track training performance:
    tr_batch_count = 0
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    #num_batch = tr_set[0].shape[0]
    for i, data in enumerate(trDataloader):
        
    #for data in tqdm(trDataloader):
    #for i in tqdm(range(num_batch)):
        hist, mask, lat_enc, lon_enc, fut, op_mask = data
        

        if args['use_cuda']:
            hist = hist.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        # Forward pass
        if args['use_maneuvers']:
            #fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred = net(hist, lat_enc, lon_enc)
            # Pre-train with MSE loss to speed up training
            if epoch< pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask)# + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                #avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                #avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, lat_enc, lon_enc)
            if epoch< pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        #a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        #optimizer.module.step()

        # Track average train loss and average train time:
        avg_tr_loss += l.item()
        tr_batch_count += 1
        
    epoch_loss = avg_tr_loss/tr_batch_count
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    print('epoch:{},avg_loss:{}'.format(epoch, epoch_loss))
        #save best model
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        torch.save({'epoch': epoch, 'state_dict': net.state_dict()},
                           'output/sunday/model_{}.pth'.format(epoch))
        print("epoch:%d Model Saved" % epoch)
        print("epoch: {} complete; Loss: {}; Time taken(s):{}".format(epoch, epoch_loss, time.time()-st))

    
    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False
    epoch_num = epoch
    print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i,data in enumerate(valDataloader):
    #for i in range(val_set[0].shape[0]):
        st_time = time.time()
        hist, mask, lat_enc, lon_enc, fut, op_mask = data

        if args['use_cuda']:
            hist = hist.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        # Forward pass
        if args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _ , _ = net(hist, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, lat_enc, lon_enc)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1

    avg_val_loss = avg_val_loss/val_batch_count
    print(avg_val_loss)
    
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('val lat acc', avg_val_lat_acc/val_batch_count*100, epoch)
    writer.add_scalar('val lon acc', avg_val_lon_acc/val_batch_count*100, epoch)
    total_acc = (avg_val_lat_acc/val_batch_count)*(avg_val_lon_acc/val_batch_count)*100
    writer.add_scalar('val total acc', total_acc, epoch)
    # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count

writer.close()
    



    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#torch.save(net.state_dict(), 'trained_models/cslstm_m.tar')



