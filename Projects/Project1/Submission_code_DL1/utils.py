# -*- coding: utf-8 -*-
"""
PROJECT 1 - DEEP LEARNING
utils.py

Authors: Lucia Montero Sanchis, Ada Pozo and Milica Novakovic

Helper functions to train the models
"""


import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC # "Support vector classifier"



#############################################################
#                   LSTM training functions                 #
#############################################################

def train_model_LSTM(model, train_loader, test_loader, num_epochs=150,
    lr=.001, verbose = True, weight_decay = 0, force_cpu=False):

    '''
    Train and test the LSTM model

    PARAMETERS
        model - LSTM model
        train_loader - pytorch loader of the train set
        test_loader - pytorch loader of the test set
        num_epochs - number of epochs to train the model
        lr - learning rate
        verbose - print progress every epoch
        weight_decay - L2 regularization
        force_cpu - use cpu even with gpu available
    OUTPUT
        Vector with training loss per epoch
        Vector with training accuracy per epoch
        Vector with validation loss per epoch
        Vector with validation accuracy per epoch
    '''
    # Save accuracies
    train_epoch_loss = []
    test_epoch_loss = []
    train_epoch_acc = []
    test_epoch_acc = []

    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Move to GPU
    if torch.cuda.is_available() and not force_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Iterate through epochs
    for epoch in range(0, num_epochs):

        # train for one epoch
        train_loss, train_acc, nb_total = train_LSTM(train_loader, model, criterion,
            optimizer, force_cpu=force_cpu)
        # normalize outputs
        train_acc = train_acc/nb_total
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)

        # evaluate on validation set
        test_loss, test_acc, nb_total_test = validate_LSTM(test_loader, model,
            criterion, force_cpu=force_cpu)

        # Normalize outputs
        test_acc = test_acc / nb_total_test
        test_epoch_loss.append(test_loss)
        test_epoch_acc.append(test_acc)

        # Print progress
        if verbose:
            print ('Epoch [%d/%d]:\n            Train Loss: %.4f, Train acc %.4f\n            Test Loss: %.4f, Test acc: %.4f \n'
             %(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))


    return  train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc

def train_LSTM(train_loader, model, criterion, optimizer, force_cpu=False):
    '''
    Train the LSTM model

    PARAMETERS
        model - LSTM model
        train_loader - pytorch loader of the train set
        criterion - loss function
        optimizer - optimizer function
        force_cpu - use cpu even with gpu available
    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for training, in total
    '''
    # Change to train mode
    model.train()
    nb_correct = 0  # Number of correct
    nb_total = 0    # Total number of samples
    loss_epoch = [] # Loss

    #Iterate through minibatches
    for i, (inputs, target) in enumerate(train_loader):
        # Check that seq_len is the same as number of timesteps
        assert (model.seq_len==inputs.shape[2]), 'seq_len in model must be equal to the number of timesteps in data'

        # Permute so they are in the form minibatch x timesteps x channels
        X = torch.autograd.Variable(inputs.view(model.minibatch, model.seq_len, -1))
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()


        # Clear gradients
        model.zero_grad()

        # Clear hidden state
        model.hidden = model.init_hidden()

        # Forward pass
        scores = model(X)

        # Predicted label (highest probability)
        pred_label = scores.data.max(1)[1]

        #Compute loss
        loss = criterion(scores, y.long())
        loss_epoch.append(loss.data)
        nb_correct += (pred_label==y.data).sum()
        nb_total += len(pred_label)

        #Backward pass
        loss.backward()
        optimizer.step()


    return np.mean(loss_epoch), nb_correct, nb_total

def validate_LSTM(test_loader, model, criterion, force_cpu=False):
    '''
    Validate the LSTM model

    PARAMETERS
        model - LSTM model
        test_loader - pytorch loader of the test set
        criterion - loss function
        force_cpu - use cpu even with gpu available
    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for validation, in total
    '''
    # switch to evaluate mode
    model.eval()

    nb_correct = 0 # Number of correct
    nb_total = 0   # Total number of samples
    test_loss = [] # Loss

    #Iterate through minibatches
    for i, (inputs, target) in enumerate(test_loader):
        # Check that seq_len is the same as number of timesteps
        assert (model.seq_len==inputs.shape[2]), 'seq_len in model must be equal to the number of timesteps in data'

        # Permute so they are in the form minibatch x timesteps x channels
        X = torch.autograd.Variable(inputs.view(model.minibatch, model.seq_len, -1))
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()

        # Obtain predictions
        scores = model(X)
        # Predicted label (highest probability)
        pred_label = scores.data.max(1)[1]

        # Loss
        loss = criterion(scores, y.long())
        test_loss.append(loss.data)

        # Accuracy
        nb_correct += (pred_label==y.data).sum()
        nb_total += len(pred_label)


    return np.mean(test_loss), nb_correct, nb_total


#############################################################
#                    CNN training functions                 #
#############################################################

def train_model_CNN(model, train_loader, test_loader, num_epochs=10,
    verbose=True, weight_decay=0, lr=1e-3, force_cpu=False):
    """ Carries out the training and validation for a CNN model,
    based on the functions train_CNN and validate_CNN.
    INPUT
        model: CNN model
        train_loader: DataLoader object to load train data
        test_loader: DataLoader object to load test data
        num_epochs: number of epochs for the training
        verbose: if True, prints traint and validation information
        weight_decay: Weight decay for Adam optimizer (L2 penalty)
        lr: Learning rate for Adam optimizer
    OUTPUT
        Vector with training loss per epoch
        Vector with training accuracy per epoch
        Vector with validation loss per epoch
        Vector with validation accuracy per epoch
    """

    # Vectors for keeping the results
    train_e_loss = []
    test_e_loss = []
    train_e_acc = []
    test_e_acc = []

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)

    if torch.cuda.is_available() and not force_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Iterate over epochs
    for epoch in range(0, num_epochs):
        # Train (for one epoch)
        train_loss, train_acc, train_tot = train_CNN(train_loader, model,
            criterion, optimizer)

        train_loss /= train_tot
        train_acc /= train_tot

        train_e_loss.append(train_loss)
        train_e_acc.append(train_acc)

        # Evaluate
        test_loss, test_acc, test_tot = validate_CNN(test_loader, model,
            criterion)

        test_loss /= test_tot
        test_acc /= test_tot

        test_e_loss.append(test_loss)
        test_e_acc.append(test_acc)

        # Print results
        if verbose:
            print ('Epoch [%d/%d]:\n            Train Loss: %.4f, Train acc %.4f\n            Test Loss: %.4f, Test acc: %.4f \n'
            %(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))


    return train_e_loss, train_e_acc, test_e_loss, test_e_acc


def train_CNN(train_loader, model, criterion, optimizer, force_cpu=False):
    """ Carries out the training for a CNN model
    INPUT
        train_loader: DataLoader object to load train data
        model: CNN model
        criterion: criterion used to compute the loss
        optimizer: optimizer to be used for the training of the network
        force_cpu - use cpu even with gpu available
    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for training, in total
    """
    # Set model for training
    model.train()

    # Initialize counters to 0
    nb_correct = 0
    nb_total = 0
    loss_epoch = 0

    # Iterate over batches
    for i, (inputs, target) in enumerate(train_loader):
        # Create Variable
        X = torch.autograd.Variable(inputs)
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()

        # Clear gradients
        model.zero_grad()

        # Forward pass
        scores = model(X)

        # Predicted labels (the one with highest probability)
        pred_label = scores.data.max(1)[1]

        # Compute and store the loss
        loss = criterion(scores, y)
        loss_epoch += loss.data[0]

        # Update nb. correct and nb. total
        nb_correct += (pred_label == y.data).sum()
        nb_total += len(pred_label)

        # Backward pass
        loss.backward()
        optimizer.step()

    return loss_epoch, nb_correct, nb_total

def validate_CNN(test_loader, model, criterion, force_cpu=False):
    """ Carries out the validation for a trained CNN model
    INPUT
        test_loader: DataLoader object to load test data
        model: trained CNN model
        criterion: criterion used to compute the loss
        force_cpu - use cpu even with gpu available
    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for validation, in total
    """
    # Switch to evaluate mode
    model.eval()

    # Initialize counters
    nb_correct = 0
    nb_total = 0
    test_loss = 0

    # Iterate over batches
    for i, (inputs, target) in enumerate(test_loader):
        X = torch.autograd.Variable(inputs)
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()

        # Obtain predictions
        scores = model(X)

        # Predicted label (highest probability)
        pred_label = scores.data.max(1)[1]

        # Loss
        loss = criterion(scores, y)
        test_loss += loss.data[0]

        # Update nb. correct and nb. total
        nb_correct += (pred_label == y.data).sum()
        nb_total += len(pred_label)

    return test_loss, nb_correct, nb_total
    
    
#############################################################
#                 CNN + LSTM training functions             #
#############################################################
def train_model_CNN_LSTM(model, train_loader, test_loader, num_epochs=200, 
    lr=.001, verbose = True, weight_decay=0.03162, force_cpu=False):
    '''
    Train and test the LSTM + CNN model

    PARAMETERS
        model - LSTM model
        train_loader - pytorch loader of the train set
        test_loader - pytorch loader of the test set
        num_epochs - number of epochs to train the model
        lr - learning rate
        verbose - print progress every epoch
        weight_decay - L2 regularization
        force_cpu - use cpu even with gpu available
    OUTPUT
        Vector with training loss per epoch
        Vector with training accuracy per epoch
        Vector with validation loss per epoch
        Vector with validation accuracy per epoch
    '''
    train_epoch_loss = []
    test_epoch_loss = []
    train_epoch_acc = []
    test_epoch_acc = []

    # Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Move to GPU
    if torch.cuda.is_available() and not force_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Iterate trough epochs
    for epoch in range(0, num_epochs):

        # train for one epoch
        train_loss, train_acc, nb_total = train_LSTM_CNN(train_loader, model, criterion,
            optimizer, force_cpu=force_cpu)
        train_acc /= nb_total
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)


        # evaluate on validation set
        test_loss, test_acc, nb_total = validate_LSTM_CNN(test_loader, model, criterion,
            force_cpu=force_cpu)
        test_acc /= nb_total
        test_epoch_loss.append(test_loss)
        test_epoch_acc.append(test_acc)


        if verbose:

            print ('Epoch [%d/%d]:\n            Train Loss: %.4f, Train acc %.4f\n            Test Loss: %.4f, Test acc: %.4f \n'
            %(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))


    return  train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc

def train_LSTM_CNN(train_loader, model, criterion, optimizer, force_cpu=False):
    '''
    Train and test the LSTM + CNN model
    PARAMETERS
        model - LSTM model
        train_loader - pytorch loader of the train set
        criterion - loss function
        optimizer - optimizer function
        force_cpu - use cpu even with gpu available

    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for training, in total

    '''
    #Switch to train
    model.train()
    nb_correct = 0
    loss_epoch = []
    nb_total =0

    #Iterate through minibatches
    for i, (inputs, target) in enumerate(train_loader):


        X = torch.autograd.Variable(inputs)
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()

        # Clear gradients
        model.zero_grad()

        # Clear hidden state
        model.hidden = model.init_hidden()

        # Forward pass
        scores = model(X)

        # Predicted label (highest probability)
        pred_label = scores.data.max(1)[1]

        #Compute loss
        loss = criterion(scores, y.long())
        loss_epoch.append(loss.data)
        nb_correct += (pred_label==y.data).sum()
        nb_total += len(pred_label)

        #Backward pass
        loss.backward()
        optimizer.step()


    return np.mean(loss_epoch), nb_correct, nb_total

def validate_LSTM_CNN(test_loader, model, criterion, force_cpu=False):
    '''
    Validate the LSTM+CNN model

    PARAMETERS
        model - LSTM + CNN model
        test_loader - pytorch loader of the test set
        criterion - loss function
        force_cpu - use cpu even with gpu available
    OUTPUT
        array containing the loss per epoch
        array containing the correct number of predictions per epoch
        number of samples used for validation, in total
    '''
    # switch to evaluate mode
    model.eval()

    nb_correct = 0
    nb_total = 0
    test_loss = []

    #Iterate through minibatches
    for i, (inputs, target) in enumerate(test_loader):
        X = torch.autograd.Variable(inputs)
        y = torch.autograd.Variable(target)

        # Move to GPU
        if torch.cuda.is_available() and not force_cpu:
            X = X.cuda()
            y = y.cuda()

        # Obtain predictions
        scores = model(X)
        # Predicted label (highest probability)
        pred_label = scores.data.max(1)[1]

        # Loss
        loss = criterion(scores, y.long())
        test_loss.append(loss.data)

        nb_correct += (pred_label==y.data).sum()
        nb_total+= len(pred_label)
    return np.mean(test_loss), nb_correct, nb_total



#############################################################
#                     SVM training functions                #
#############################################################
def reshape_samples(data):
    nsamples, nx, ny = data.shape  
    return data.view(nsamples, nx * ny)

def train_SVM(model, train_data, test_data):

    # Get train/test input and target
    train_input = train_data.input
    train_target = train_data.target
    
    test_input = test_data.input
    test_target = test_data.target
    
    # Reshape so the we have one row per sample
    train_input_reshaped = reshape_samples(train_input).cpu()
    test_input_reshaped = reshape_samples(test_input).cpu()
    

    
    # Train model
    model.fit(train_input_reshaped, train_target)
    
    # Get scores
    train_score = model.score(train_input_reshaped, train_target) 
    test_score = model.score(test_input_reshaped, test_target)
    
    return train_score, test_score