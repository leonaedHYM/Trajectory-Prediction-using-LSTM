from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np

import time
import torch.nn.functional as F
from utils import ngsimDataset



batch_size = 64
data_dir = '/mnt/e/Northwestern University/Courses/2022winter/pattern recognition/projects/data/TrainSet.mat'
    
Set = ngsimDataset(data_dir)
Dataloader = DataLoader(Set,1,shuffle=True,collate_fn=Set.collate_fn)
dataset = torch.zeros(1,16,batch_size,14)
batch_data = torch.zeros(16,batch_size,14)
batch_lat = torch.zeros(batch_size,3)
batch_lon = torch.zeros(batch_size,2)
batch_fut = torch.zeros(25,batch_size,2)
batch_opmask = torch.zeros(25,batch_size,2)
for i, data in enumerate(Dataloader):
    
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
   
    # padding for batchsize = 1:
    
    if nbrs.shape[1] >= 6:
        for j in range(6):
            hist = torch.cat((hist, nbrs[:,j,:].unsqueeze(dim=1)),2)
    if nbrs.shape[1] < 6:
        
        hist = torch.cat((hist, nbrs.view(16,1,-1)),2)
        pad = (0,2 * (6 - nbrs.shape[1]))
        hist = F.pad(hist, pad, "constant", 0)
            
    # batching:
    if i%batch_size == 0 and i != 0:
        dataset = torch.cat((dataset, batch_data.unsqueeze(dim=0)),0)
        
        lat_set = torch.cat((lat_set, batch_lat.unsqueeze(dim=0)),0)
        lon_set = torch.cat((lon_set, batch_lon.unsqueeze(dim=0)),0)
        fut_set = torch.cat((fut_set, batch_fut.unsqueeze(dim=0)),0)
        opmask_set = torch.cat((opmask_set, batch_opmask.unsqueeze(dim=0)),0)
        
    idex = i%batch_size
    batch_data[:,idex, :] = hist.squeeze()   # 每个batch都一样
    batch_lat[idex] = lat_enc.squeeze()
    batch_lon[idex] = lon_enc.squeeze()
    batch_fut[:,idex, :] = fut.squeeze()
    batch_opmask[:,idex, :] = op_mask.squeeze()
    
    if i ==batch_size-1:
        dataset[0] = batch_data
        lat_set = batch_lat.view(1,batch_size, 3)
        lon_set = batch_lon.view(1,batch_size, 2)
        fut_set =  batch_fut.view(1,25,batch_size,2)
        opmask_set = batch_opmask.view(1,25,batch_size,2)

    if i == batch_size:
        data_i = batch_data
    
    if i == 5 * batch_size:
        data_5i = batch_data
        if torch.equal(data_i, data_5i):
            print('wrong')
        else:
            print('right') 
            
output = (dataset, lat_set, lon_set,fut_set, opmask_set)
torch.save(output, 'testset_batchSize='+str(batch_size)+'.pt')    



class ngsimDataset(Dataset):
    
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid


    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int) # traj的第一维度
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)  # get 待测车的 traj
        fut = self.getFuture(vehId,t,dsId)          # get 待测车的 未来轨迹

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,lat_enc,lon_enc

    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist


    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        #for hist,_,nbrs,_,_ in samples:    #samples 表示一个batch大小
            
            #hist = torch.from_numpy(hist)   # （16， 2）
            #for id,nbr in enumerate(nbrs):
                #it = 0
                #if it <= 6:
                    #if len(nbr)!=0:
                        #nbr = torch.from_numpy(nbr)     #（16，2）
                        #hist = torch.cat((hist,nbr),1)  # 拼接ego和nbrs
                #it += 1
            #if hist.shape[1] < 14:
                #p1d = (0,14-hist.shape[1])
                #hist = F.pad(hist, p1d, "constant", 0)

        maxlen = self.t_h//self.d_s + 1
        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),14)    #加入neighbours后变为14
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        

        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):
            hist = torch.from_numpy(hist)   # （16， 2）
            it = 0
            for id,nbr in enumerate(nbrs):
                
                if it <= 6:
                    if len(nbr)!=0:
                        nbr = torch.from_numpy(nbr)     #（16，2）
                        hist = torch.cat((hist,nbr),1)  # 拼接ego和nbrs
                        it += 1
            if hist.shape[1] < 14:
                p1d = (0,14-hist.shape[1])
                hist = F.pad(hist, p1d, "constant", 0)
                
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            for k in range(14):
                hist_batch[0:hist.shape[0],sampleId,k] = hist[:, k]
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            
            

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1

        return hist_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch