# -*- coding: utf-8 -*-
"""
PROJECT 1 - DEEP LEARNING
preprocessing.py

Authors: Lucia Montero Sanchis, Ada Pozo and Milica Novakovic

Preprocessing helper functions
"""


from scipy import signal
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import numpy as np

def preprocess_filter(data, fs=100):
    '''
    Apply the notch filters and the high pass filters to one sample of the data. 
    The notch filter has frequency 50 Hz and quality factor 30.
    The highpass filter has cutoff frequecy 0.5 Hz.

    INPUT:
        data - one sample with the format channels x time
        fs - sampling frequency. Default: 100
    OUTPUT:
        data after applying the filters with the same format

    '''
    #Notch filter
    fnotch = 50 #Frequency to be removed
    Q = 30 #Quality factor
    b_notch, a_notch = signal.iirnotch(fnotch/(fs/2), Q) #Create filter normalising the frequency

    #High pass: 3rd order, IIR butterworth filter, f0=0.5Hz
    fhigh=0.5
    b_high, a_high=signal.butter(3, fhigh/(fs/2), btype='high')


    #Apply the filters
    data=signal.lfilter(b_notch,a_notch,data)
    data=signal.lfilter(b_high,a_high,data)

    return data
 

def preprocessing(data, ignore_outliers=False, fs = 100):
    '''
    Preprocessing of the data
    INPUT:  
        data - array with the data 
        ignore_outliers - if True, use RobustScaler. Otherwise, StandardScaler. Default: False
        fs - sampling frequency. Default: 100
    OUTPUT:
        torch tensor with preprocessed data
    '''
    #Use RobustScaler
    if ignore_outliers:
        scaler = RobustScaler()
    #Use normal standardization
    else:
        scaler = StandardScaler()

    # If a 3d array is given, reshape so we can apply the scalers
    reshape = None
    if np.ndim(data)>2:
        reshape = data.shape
        data = reshape_samples(data)
        
    # Apply standarization
    data = scaler.fit_transform(data)

    # Apply filtering
    data = preprocess_filter(data, fs)
    
    # If reshaping was done, reshape it back to original dimensions
    if reshape is not None:
        data = np.reshape(data, reshape)

    
    return torch.FloatTensor(data)
    

def reshape_samples(data):
    '''
    Reshape samples from 3D to 2D
    '''
    nsamples, nx, ny = data.shape  
    return data.view(nsamples, nx * ny)
    
 