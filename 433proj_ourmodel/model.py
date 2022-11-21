from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation

class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, 64)
        self.hidden_size = 128
        self.num_layers = 1

        # define LSTM layer
        self.lstm = nn.LSTM(64, hidden_size=128,  # input æ˜?(batch_size, sequence_size, input_size)
                            num_layers=1, batch_first=False)

        # define activation:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x_input):
        embedded = self.embedding(x_input)
        embedded = self.leaky_relu(embedded)
        lstm_out, hidden = self.lstm(embedded)

        return lstm_out, hidden

class highwayNet(nn.Module):

    ## Initialization
    def __init__(self,args, enc_input_size, hidden_size=128, output_timestep = 25, output_size = 2):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        self.enc_input_size = enc_input_size
        self.encoder = lstm_encoder(self.enc_input_size)
        self.hidden_size = hidden_size

        ## Sizes of network layers
        self.in_length = args['in_length']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        # Decoder LSTM
        if self.use_maneuvers:
            #self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            self.dec_lstm = torch.nn.LSTM(self.hidden_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)

        else:
            #self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)
            self.dec_lstm = torch.nn.LSTM(self.hidden_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        self.op_lat = torch.nn.Linear(self.hidden_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.hidden_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    ## Forward Pass
    def forward(self,hist, lat_enc,lon_enc):
        input_batch = hist  
        _, enc = self.encoder(input_batch)
        #print(enc.shape)
        #enc = torch.squeeze(enc[0])
        enc=enc[0][0]
        #encodeing shape: (32,112)

        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)  #(32,117)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc) #(25,32,5)
            return fut_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)  #(25,32,112)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)  #(32,25,128)
        fut_pred = self.op(h_dec)  #(32,25,5)
        fut_pred = fut_pred.permute(1, 0, 2) #(25,32,5)
        fut_pred = outputActivation(fut_pred) #(25,32,5)
        return fut_pred





