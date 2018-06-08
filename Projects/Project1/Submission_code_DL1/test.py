# -*- coding: utf-8 -*-
"""

PROJECT 1 - DEEP LEARNING
test.py


Lucia Montero Sanchis, Ada Pozo and Milica Novakovic

Obtain the results provided in the report
"""


from models import *
from bci_dataset import *
import torch
from utils import *
from sklearn.svm import SVC # "Support vector classifier"

def run_test_CNN(train_loader, test_loader, mb, seq_len = 50, 
    verbose = False, num_epochs=-1,  force_cpu = False):
    nb_hidden = 100
    conv1 = 200
    ks1 = 10
    conv2 = 125
    ks2 = 10
    p_drop = 0.0316
    weight_decay = 0.0316
    lr = 0.001

    # If number of epochs not specified, run default
    if num_epochs == -1:
        num_epochs = 200
    
    # Create model
    model = ShallowCNN(seq_len, mb, nb_hidden=nb_hidden,
        conv1=conv1, ks1=ks1, conv2=conv2, ks2=ks2, p_drop=p_drop)

    print('\nTraining CNN model for {} epochs'.format(num_epochs))
    train_loss, train_acc, test_loss, test_acc = train_model_CNN(model,
        train_loader, test_loader, verbose=verbose, num_epochs=num_epochs,
        weight_decay=weight_decay, lr=lr,  force_cpu = force_cpu)

    print("    > Results for CNN:")
    print("    >    Train accuracy in last epoch: ", train_acc[-1])
    print("    >    Test accuracy in last epoch: ", test_acc[-1])

def run_test_LSTM(train_loader, test_loader, minibatch, seq_len = 50, 
    verbose = False, num_epochs=-1,  force_cpu = False):
    num_layers = 1
    hidden_dim = 128
    dropout = 0.5
    weight_decay = 0
    lr = 0.001
    
   

    # If number of epochs not specified, run default
    if num_epochs == -1:
        num_epochs = 60
    # Create model
    model = LSTM(minibatch = minibatch, seq_len = train_dataset.input.shape[2], 
        hidden_dim = hidden_dim, num_layers = num_layers, dropout = dropout)
    
    # Train
    print('\nTraining LSTM model for {} epochs'.format(num_epochs))
    train_loss, train_acc, test_loss, test_acc = train_model_LSTM(model, 
        train_loader, test_loader, verbose=verbose, num_epochs=num_epochs,
        weight_decay = weight_decay,  lr=lr,  force_cpu = force_cpu)


    print("    > Results for LSTM:")
    print("    >    Train accuracy in last epoch: ", train_acc[-1])
    print("    >    Test accuracy in last epoch: ", test_acc[-1])
    
    
def run_test_LSTM_CNN(train_loader, test_loader, minibatch, seq_len = 50,
    verbose = False, num_epochs=-1, force_cpu = False):
    #General params
    weight_decay = 0.0316
    lr = 0.001
    
    #LSTM params
    num_layers_LSTM = 1
    hidden_dim_LSTM = 128
    dropout_LSTM = 0.5
    
    #CNN params
    nb_hidden = 100
    conv1 = 200
    ks1 = 10
    conv2 = 125
    ks2 = 10
    dropout_CNN = 0.0316

    # If number of epochs not specified, run default
    if (num_epochs == -1):
        num_epochs = 200
    
    # Create model
    model = LSTM_CNN(nb_hidden=nb_hidden, conv1=conv1, ks1=10, 
        conv2=conv2, ks2=ks2, dropout_CNN=dropout_CNN, 
        minibatch = minibatch, num_layers_LSTM = num_layers_LSTM, 
        seq_len = train_dataset.input.shape[2], 
        hidden_dim_LSTM = hidden_dim_LSTM, dropout_LSTM = dropout_LSTM)
    
    print('\nTraining combined LSTM + CNN model for {} epochs'.format(num_epochs))
    train_loss, train_acc, test_loss, test_acc = train_model_CNN_LSTM(model, 
        train_loader, test_loader, verbose=verbose, num_epochs=num_epochs,
        weight_decay = weight_decay,  lr=lr,  force_cpu = force_cpu)


    print("    > Results for combined model of LSTM and CNN:")
    print("    >    Train accuracy in last epoch: ", train_acc[-1])
    print("    >    Test accuracy in last epoch: ", test_acc[-1])

def run_test_SVM(train_dataset, test_dataset, kernel, gamma, C):
    
    # Create model
    model = SVC(kernel=kernel, gamma=gamma, C=C)
    
    # Train
    print('\nTraining SVM ')
    train_acc, test_acc = train_SVM(model, train_dataset, test_dataset)


    print("    > Results for SVM:")
    print("    >    Train accuracy: ", train_acc)
    print("    >    Test accuracy: ", test_acc)

import argparse
import os

######################################################################

parser = argparse.ArgumentParser(description='''DEEP LEARNING PROJECT 1: Per default run with the 
                                            same parameters and number of epochs as in the report   
                                            with the downsampled version of the dataset and no 
                                            filtering.
                                            
                                            \n If you only want to run the final model (combined 
                                            LSTM and CNN use): 
                                            \n     "python test.py --only_final". 
                                            
                                            \n If you want to run for only n epochs all neural nets models use: 
                                            \n     "python test.py --num_epochs n".
                                            
                                            \n If you want to run the models with preprocessed data use: 
                                            \n     "python test.py --filter". 
                                            
                                            \n If you want to run the models with data with  1 kHz sampling rate use:
                                            \n     "python test.py --one_khz\n") 
                                            ''')

parser.add_argument('--num_epochs',
                    type=int, default=-1,
                    help = 'Number of epochs to run. If -1, use the same as in the report \
                    - may take a long time (default -1)')

parser.add_argument('--mb',
                    type=int, default=25,
                    help = 'Minibatch size (default 25)')

parser.add_argument('--filter',
                    action='store_true', default=False,
                    help = 'Apply preprocessing before classification (default False)')
                    

parser.add_argument('--one_khz',
                    action='store_true', default=False,
                    help = 'Use data with 1 kHz sampling rate (default False)')    

parser.add_argument('--verbose',
                    action='store_true', default=False,
                    help = 'Print results per epoch (default False)')                        

parser.add_argument('--force_cpu',
                    action='store_true', default=False,
                    help = 'Keep tensors on the CPU, even if cuda is available (default False)')

parser.add_argument('--only_final',
                    action='store_true', default=False,
                    help = 'Run only final model (combined LSTM and CNN) (default False)')
                                        
parser.add_argument('--kernel',
                    type = str, default = 'linear',
                    help = 'kernel (linear or rbf) value for the SVM (default linear)') 
                    
parser.add_argument('--gamma',
                    type = float, default = 1e-5,
                    help = '\gamma value for the SVM (ignored by linear kernel) (default 1e-5)')


parser.add_argument('--C',
                    type = float, default = 0.0001,
                    help = 'C (regularization) value for the SVM (default 0.0001)')


args = parser.parse_args()


if torch.cuda.is_available() and not args.force_cpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

print(' \n############# DEEP LEARNING  - PROJECT 1 ###############')
print('This script obtains the same results as in the report \
with 100 Hz sampling rate and no filtering. It runs all models \n\
(SVM, CNN, LSTM, CNN+LSTM). It trains for the neural nets \
for the same number of epochs as shown in the report, \n \
which on a CPU make take longer than 20 minutes. \
\n If you only want to run the final model (combined \
LSTM and CNN use): \
\n     python test.py --only_final \
\n If you want to run for only n epochs all neural net models use: \
\n     python test.py --num_epochs n \
\n If you want to run the models with preprocessed data use: \
\n     python test.py --filter \
\n If you want to run the models with data with  1 kHz sampling rate use: \
\n     python test.py --one_khz\n') 
    
# Load training dataset
print('Loading dataset')
train_dataset = bci(train=True, one_khz=args.one_khz, filter = args.filter, 
                    force_cpu = args.force_cpu)
test_dataset = bci(train=False, one_khz=args.one_khz, filter = args.filter,
                    force_cpu = args.force_cpu)

seq_len = train_dataset.input.shape[2]

train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.mb, drop_last=True,
            shuffle = True)

test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=args.mb, drop_last=True)

# Run models

if args.only_final == False:
    print(' \n################ SVM ##################')
    run_test_SVM(train_dataset, test_dataset, args.kernel, args.gamma, args.C)

    print(' \n################ LSTM #################')
    run_test_LSTM(train_loader, test_loader, args.mb, seq_len, verbose=args.verbose,
                    num_epochs = args.num_epochs,  force_cpu = args.force_cpu)

    print(' \n################ CNN ##################')
    run_test_CNN(train_loader, test_loader, args.mb, seq_len, verbose=args.verbose,
                num_epochs = args.num_epochs,  force_cpu = args.force_cpu)

print(' \n############# LSTM + CNN ##############')
run_test_LSTM_CNN(train_loader, test_loader, args.mb, seq_len, verbose=args.verbose,
                num_epochs = args.num_epochs,  force_cpu = args.force_cpu)