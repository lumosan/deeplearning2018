# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


################ Generate data ################
def generate_disc_set(nb):
    """Generate dataset

    INPUT
        nb: number of points to generate
    OUTPUT:
        data
        labels with one hot encoding
    """
    # Create nb samples between 0 and 1
    data = Tensor(nb, 2).uniform_(0, 1)
    # Points inside the circle with radius 1/2 centered in 0 have label 1, otherwise 0
    label = ((data - .5) ** 2).sum(1) <= 1 / (2 * np.pi)
    return data, convert_to_one_hot_labels(data, label.long())


def convert_to_one_hot_labels(input_, target):
    """Convert labels to one-hot encoding
    Function taken from the course prologue
    """
    tmp = input_.new(target.size(0), max(0, target.max()) + 1).fill_(0)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp.long()



################ Auxiliary functions ################
def plot_points(input_, target, pred=None, alpha=.5, highlight_errors=True,
    errors_color="red", title=None):
    """Scatter plot of the classes in dataset
    INPUT
        input_: data
        target: labels
        pred: estimated classes of datapoints
        alpha: transparency of points in plot
        highlight_errors: plot the errors in prediction
        errors_color: color for the errors in prediction
    """
    plt.figure()
    if title is not None:
        plt.title(title)

    # Cannot plot errors without prediction
    if highlight_errors:
        assert pred is not None
    # Samples from each class
    input_0 = input_[target[:,0].nonzero(),:].view(-1,2)
    input_1 = input_[target[:,1].nonzero(),:].view(-1,2)
    # Plot
    plt.scatter(input_0[:,0], input_0[:,1], c="gray", alpha=1, label = "Class 0")
    plt.scatter(input_1[:,0], input_1[:,1], c="lightgray", alpha=1, label = "Class 1")
    # Show errors
    if highlight_errors:
        # Indexes of incorrect labeled points
        idx = (pred != target[:,0]).nonzero()
        # Plot if there are errors
        if len(idx.shape):
            errors = input_[idx,:].view(-1,2)
            plt.scatter(errors[:,0], errors[:,1], c=errors_color, alpha=alpha, label="Errors")
    plt.legend()
    plt.show()


def compute_labels(predicted):
    """Compute the labels of the prediction
    INPUT:
        predicted: prediction with one-hot encoding
    OUTPUT
        predicted labels with one hot encoding
    """
    # Class with biggest probability is the predicted
    res = torch.max(predicted, 1, keepdim=False, out=None)[1]
    # Convert to one hot labels
    lbl = convert_to_one_hot_labels(Tensor(), res)
    return lbl

def train_test_split(X,y,split=0.2):
    '''
    Divide data in train and test sets
    
    INPUT:
      X - train input
      y - train target
      split - proportion of test samples
    OUTPUT:
      trainX, testX, trainY, testY
    
    '''  
    #Permute
    X = X[torch.randperm(len(X))]
    y = y[torch.randperm(len(y))]
    #Number of test samples
    test_samples=int(split*(len(X)))
    testX, testY = X[:test_samples],y[:test_samples]
    trainX, trainY  = X[test_samples:],y[test_samples:]
      
    return trainX,testX,trainY,testY

################ Training and testing functions ################


def test(model, loss, test_input, test_target, verbose = True):
    """Test the model

    INPUT
        model
        loss: loss function
        test_input: input
        test_target: labels
        verbose: if True, write accuracy
    OUTPUT
        accuracy
    """
    model.train = False
    # Get output and loss
    output = model.forward(test_input)
    L = loss.forward(output, test_target)
    # Get predicted labels
    labels = compute_labels(output)[:,0]
    # Compute accuracy
    errors = (test_target[:,0] != labels).sum()
    accuracy = (len(test_target) - errors) / len(test_target)

    if verbose:
        print(" >>>   Test: Loss {:.08f}  Accuracy {:.02f}  Errors {}".format(
            L, accuracy, errors))

    return accuracy, labels


def predict(model, input_):
    """Get prediction

    INPUT
        model
        input_

    OUTPUT
        output of the model given the input
    """
    # Get output
    output = model.forward(input_)
    # Get predicted labels
    labels = compute_labels(output)[:,0]

    return labels


def train(optimizer, model, loss, n_epochs, mini_batch_size,
    train_input, train_target, test_input, test_target, verbose=True):
    """Train the model without early stopping

    INPUT
        optimizer
        model
        loss: loss function
        n_epochs: number of epoch to run
        mini_batch_size
        train_input: input
        train_target: labels
        verbose: if True, write accuracy and loss per epoch
    OUTPUT
        accuracy per epoch and final
    """

    output_vals = Tensor()
    max_range = None

    # If mini_batch_size is a multiple of the number of samples, use them all
    if train_input.size(0) % mini_batch_size == 0:
        max_range = train_input.size(0)
    # If not, last samples are not used
    else:
        max_range = train_input.size(0) - mini_batch_size
    acc_train = []
    acc_test = []
    #Iterate through epochs
    for e in range(n_epochs):
        model.train = True
        # Variables for loss, number of errors and prediction
        L_tot = 0
        errors_tot = 0
        pred_acc = Tensor().long()
        
        #Iterate through minibatches
        for b in range(0, max_range, mini_batch_size):
            #Input data
            d = train_input.narrow(0, b, mini_batch_size)
            #Labels
            l = train_target.narrow(0, b, mini_batch_size)

            # Forward pass
            output = model.forward(d)
            L = loss.forward(output, l)

            # Backward pass
            grad = loss.backward()
            model.backward(grad)

            #Step
            optimizer.step(model, loss)

            # Compute total loss
            L_tot += L

            # Compute metrics
            r = compute_labels(output)[:,0]
            pred_acc = torch.cat([pred_acc, r])
            errors = (l[:,0] != r).sum()
            errors_tot += errors


        # Total accuracy
        accuracy = (len(train_target) - errors_tot) / len(train_target)
        if verbose:
            print("Train: Epoch {:d}  Loss {:.08f}  Accuracy {:.02f}  Errors {}".format(
                e, L_tot, accuracy, errors_tot))
        acc_test.append(test(model, loss, test_input, test_target, verbose = verbose)[0])
        acc_train.append(accuracy)
    return accuracy, pred_acc, acc_train, acc_test
