# -*- coding: utf-8 -*-
"""
PROJECT 1 - DEEP LEARNING
bci_dataset.py

Authors: Lucia Montero Sanchis, Ada Pozo and Milica Novakovic

torch dataset to load the data
"""

from preprocessing import * 
import dlc_bci 

import torch
import torch.utils.data as data



class bci(data.Dataset):
    '''
    Creates bci dataset
    
    INPUT:
        data_path: string, path to the data. Default: ./data_bci
        train: boolean, load training or test set. Default: True
        one_khz: boolean, if false, load downsampled 100 Hz version. Default: False
        filter: boolean, if true, apply low and high pass filtering. Default: False
        robust_scaler: boolean, if true and filtering is applied, use robust scaler. If false, use StandardScaler. Default: False
        force_cpu - use cpu even with gpu available. Default: False
        num_samples: int, if crop is true, size of each window in the crop. Default: 20 (in seconds, 20/fs)
        shift: int, if crop is true, shift between windows during the cropping. Default: 10 (in seconds, 10/fs)
    OUTPUT:
        torch dataset
    '''
    
    
    def __init__(self, data_path = './data_bci', train = True, one_khz = False, filter = False, robust_scaler = False, 
                num_samples = 20, shift = 10, force_cpu = False):
        # Load data
        self.input, self.target =  dlc_bci.load(root = data_path, one_khz = one_khz, train = train)
        self.train = train
        self.force_cpu = force_cpu
        
        print('Input data loaded (size = {})'.format(self.input.shape))
        print('Target data loaded (size = {})'.format(self.target.shape))
        
        #Filtering
        if filter:
            if one_khz:
                fs = 1000
            else:
                fs = 100
            self.input = preprocessing(self.input, ignore_outliers = robust_scaler, fs = fs)
            
            if torch.cuda.is_available() and not force_cpu:
                self.input = self.input.cuda()
        

        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        sample = self.input[idx]
        target = self.target[idx]

        return sample, target
        
