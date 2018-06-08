# -*- coding: utf-8 -*-
"""
PROJECT 1 - DEEP LEARNING
models.py

Authors: Lucia Montero Sanchis, Ada Pozo and Milica Novakovic

Deep Learning models
"""


import torch
import torch.nn as nn
from bci_dataset import *

#torch.set_default_tensor_type('torch.FloatTensor')


#############################################################
#                           CNN                             #
#############################################################
class ShallowCNN(nn.Module):
    def __init__(self, ntimesteps, minibatch, nb_hidden=50,
        conv1=50, ks1=10, conv2=100, ks2=10, p_drop=0):
        ''' Create Shallow CNN model.

        PARAMETERS
            ntimesteps - Number of timesteps (50 or 500 depending on downsampling)
            minibatch - Size of the minibatch
            nb_hidden - Number of nodes in linear layer
            conv1 - Number of convolution filters in first convolutional layer
            ks1 - Kernel size in first convolutional layer
            conv2 - Number of convolution filters in second convolutional layer
            ks2 - Kernel size in first convolutional layer
            p_drop - Dropout probability
        '''
        super(ShallowCNN, self).__init__()
        self.minibatch = minibatch

        # Define first convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(28, conv1, kernel_size=ks1),
            nn.BatchNorm1d(conv1),
            nn.ReLU(),
            nn.Dropout(p=p_drop)
        )

        # Define second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=ks2),
            nn.BatchNorm1d(conv2),
            nn.ReLU(),
            nn.Dropout(p=p_drop)
        )

        # Compute the number of inputs for first linear layer
        s = ntimesteps - (ks1 - 1)
        s = s - (ks2 - 1)

        # Define first linear layer
        self.layer3 = nn.Sequential(
            nn.Linear(conv2 * s, nb_hidden),
            nn.ReLU()
        )

        # Define second linear layer
        self.layer4 = nn.Sequential(
            nn.Linear(nb_hidden, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Forward pass of backpropagation
        
        PARAMETERS
            x - input with format minibatch x channels x timesteps
        
        '''
        x = self.layer1(x)
        x = self.layer2(x)

        # Flatten inputs
        x = x.view(self.minibatch, -1)

        x = self.layer3(x)
        x = self.layer4(x)

        return x

#############################################################
#                           LSTM                            #
#############################################################

class LSTM(nn.Module):
    '''
    LSTM model using the last LSTM layer for prediction
    
    | Layer | Input              | Operation               | Output |
    |:-----:|:------------------:|:-----------------------:|:------:|
    | 1     | T x n_channels     | LTSM - 128 hidden units | 128    |
    | 1     | 128                | Dropout - 0.5           | 128    |
    | 2     | 128                | Linear + Softmax        |   2    |

    '''
    def __init__(self, minibatch = 25, num_layers = 1, seq_len=50, hidden_dim = 128, dropout = 0.5):
        '''
        Create LSTM model
        
        PARAMETERS:
            minibatch - mumber of samples in minibatch
            num_layers - number of LSTM layers
            seq_len - number of timesteps of each samples
            hidden_dim - number of hidden units of the LSTM layers
            dropout - drop probability
        
        '''
        super(LSTM, self).__init__()
        # Hidden dimension
        self.hidden_dim = hidden_dim
        # Size of minibatch
        self.minibatch = minibatch
        # Number of timesteps of sequence
        self.seq_len = seq_len
        # Number of layers
        self.num_layers = num_layers

        #LSTM: One layer, with 28 x hidden_dim. batch_first so it has format minibatch x seq_len x 28
        self.lstm =  nn.LSTM(28, hidden_dim, num_layers=num_layers,  batch_first = True)
        #Dropout
        self.d = nn.Dropout(dropout)
        

        # The linear layer that maps from hidden state space to output space (2 classes)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 2), #Output flattened  
            nn.Softmax(dim=1)
        )
        
        # Clean up the hidden statte
        self.hidden = self.init_hidden()

    def init_hidden(self):
        '''
        Create the hidden state, initializing it to zero
        '''
        # Create hidden state (num_layers, minibatch_size, hidden_dim)
        return (torch.autograd.Variable(torch.zeros(self.num_layers, self.minibatch, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(self.num_layers, self.minibatch, self.hidden_dim)))

    def forward(self, x):
        '''
        Forward pass of backpropagation with only the last LSTM
        for prediction
        
        PARAMETERS
            x - input with format minibatch x timesteps x channels
        
        '''
        # LSTM forward passing the hidden state
        out, self.hidden = self.lstm(x, self.hidden)
        # Dropout of only the last LSTM layer
        out = self.d(self.hidden[0][-1,:,:])
        # Flatten output and pass to the linear layer 
        out = self.linear(out.view(-1, self.hidden_dim))
        
        return out
        
#############################################################
#                       LSTM + CNN                          #
#############################################################        
        
class LSTM_CNN(nn.Module):
    '''
    Combination of LSTM and CNN models
    '''
    def __init__(self, nb_hidden=100,
        conv1=200, ks1=10, conv2=125, ks2=10, dropout_CNN=0.0316, minibatch = 25, num_layers_LSTM = 1, seq_len=50, 
        hidden_dim_LSTM = 128, dropout_LSTM = 0.5):
        '''
        Create CNN+LSTM model
        
        PARAMETERS:
            minibatch - number of samples in minibatch
            num_layers_LSTM - number of LSTM layers
            seq_len - number of timesteps of each samples
            hidden_dim_LSTM - number of hidden units of the LSTM layers
            dropout_LSTM - drop probability of LSTm
            nb_hidden - Number of nodes in linear layer of CNN
            conv1 - Number of convolution filters in first convolutional layer
            ks1 - Kernel size in first convolutional layer
            conv2 - Number of convolution filters in second convolutional layer
            ks2 - Kernel size in first convolutional layer
            dropout_CNN - Dropout probability of CNN model
        '''
        super(LSTM_CNN, self).__init__() 
        
        #LSTM
        # Hidden dimension
        self.hidden_dim_LSTM = hidden_dim_LSTM
        # Size of minibatch
        self.minibatch = minibatch
        # Number of timesteps of sequence
        self.seq_len = seq_len
        # Number of layers
        self.num_layers_LSTM = num_layers_LSTM

        #LSTM: One layer, with 28 x hidden_dim, dropout 0.005. batch_first so it has format minibatch x seq_len x 28
        self.lstm = nn.LSTM(28, hidden_dim_LSTM, num_layers=num_layers_LSTM, batch_first = True)
        self.d = nn.Dropout(p= dropout_LSTM)
        # Clean up the hidden state
        self.hidden = self.init_hidden()


        #CNN
        self.CNN_layer1 = nn.Sequential(
            nn.Conv1d(28, conv1, kernel_size=ks1),
            nn.BatchNorm1d(conv1),
            nn.ReLU(),
            nn.Dropout(p=dropout_CNN)
        )

        self.CNN_layer2 = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=ks2),
            nn.BatchNorm1d(conv2),
            nn.ReLU(),
            nn.Dropout(p=dropout_CNN)
        )

        s = seq_len - (ks1 - 1)
        s = s - (ks2 - 1)
        
        self.CNN_layer3 = nn.Sequential(
            nn.Linear(conv2 * s, nb_hidden),
            nn.ReLU()
        )

        
        #CONCAT
        self.fc = nn.Sequential(
            nn.Linear(nb_hidden+hidden_dim_LSTM, 2),
            nn.Sigmoid()
        )
        
        
    def init_hidden(self):
        '''
        Create the hidden state, initializing it to zero
        '''
        # Create hidden state (num_layers, minibatch_size, hidden_dim)
        return (torch.autograd.Variable(torch.zeros(self.num_layers_LSTM, 
                                                    self.minibatch, 
                                                    self.hidden_dim_LSTM)),
                torch.autograd.Variable(torch.zeros(self.num_layers_LSTM,
                                                    self.minibatch, 
                                                    self.hidden_dim_LSTM)))

    def forward(self, x):
        '''
        Forward pass of backpropagation concatenating the outputs
        of LSTM and CNN
        
        PARAMETERS
            x - input with format minibatch x channels x timesteps
        
        '''
        #CNN
        out_CNN = self.CNN_layer1(x)
        out_CNN = self.CNN_layer2(out_CNN)
        out_CNN = out_CNN.view(self.minibatch, -1)
        out_CNN = self.CNN_layer3(out_CNN)

        
        #LSTM
        out_LSTM, self.hidden = self.lstm(x.view(self.minibatch, self.seq_len, -1))
        out_LSTM = self.d(self.hidden[0])       

        #CONCAT
        out = torch.cat((out_CNN,out_LSTM.view(-1, self.hidden_dim_LSTM)),1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        
        return out
