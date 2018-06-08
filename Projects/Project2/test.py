# -*- coding: utf-8 -*-

import argparse

from framework.utils import generate_disc_set, plot_points, test, train
from framework.modules import Linear, Sequential, Dropout
from framework.activation_functions import ReLU, LeakyReLU, Tanh, Sigmoid
from framework.loss_functions import LossMSE, CrossEntropyLoss
from framework.optimizers import SGD, Adam


def run(args):
    #Generate data
    train_input, train_target = generate_disc_set(1000)
    data_0 = train_input[train_target[:,0].nonzero(),:].view(-1,2)
    data_1 = train_input[train_target[:,1].nonzero(),:].view(-1,2)

    #Plot
    plot_points(train_input, train_target, highlight_errors=False,
        title="Train dataset")

    # Model parameters
    if args.activ1 == "relu":
        activ1="ReLU()"
    elif args.activ1 == "leaky_relu":
        activ1="LeakyReLU()"
    elif args.activ1 == "tanh":
        activ1="Tanh()"
    else:
        activ1="Sigmoid()"

    if args.activ2 == "relu":
        activ2=ReLU()
    elif args.activ2 == "leaky_relu":
        activ2=LeakyReLU()
    elif args.activ2 == "tanh":
        activ2=Tanh()
    else:
        activ2=Sigmoid()

    # Generate model
    hl1 = [Linear(2, 25, lr=0), eval(activ1), Dropout(args.dropout)]
    hl2 = [Linear(25, 25, lr=0), eval(activ1), Dropout(args.dropout)]
    hl3 = [Linear(25, 30, lr=0), eval(activ1), Dropout(args.dropout)]
    out = [Linear(30, 2, lr=0), activ2]

    model = Sequential(hl1 + hl2 + hl3 + out)

    # Loss function
    if args.loss == "mse":
        loss = LossMSE()
    else:
        loss = CrossEntropyLoss()

    # Optimizer
    if args.optim == "SGD":
        k, e0= (.6, 3.5e-2)
        optim = SGD([k, e0])
    else: #Adam - default parameters
        optim = Adam()

    # Train
    tr_acc, train_pred = train(optim, model, loss, args.nepochs, args.minibatch, train_input, train_target, verbose=False)
    print("Training Accuracy: {}".format(tr_acc))
    plot_points(train_input, train_target, train_pred, highlight_errors=True,
        title="Train errors - Accuracy: {}".format(tr_acc))


    # Test
    test_input, test_target = generate_disc_set(1000)
    te_acc, test_pred = test(model, loss, test_input, test_target, verbose=False)
    print("Testing Accuracy: {}".format(te_acc))
    plot_points(test_input, test_target, test_pred, highlight_errors=True,
        title="Test errors - Accuracy: {}".format(te_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning project 2: ")

    parser.add_argument("--loss", default = "MSE", help="Loss function to use: MSE or CrossEntropy. Default: MSE", type=str)
    parser.add_argument("--optim", default = "SGD", help="Optimizer to use: SGD or Adam. Default: SGD", type=str)
    parser.add_argument("--activ1", default = "relu", help="Activation function of the first three layers: relu, leaky_relu, tanh, sigmoid. Default: relu", type=str)
    parser.add_argument("--activ2", default = "sigmoid", help="Activation function of the last layer: relu, leaky_relu, tanh, sigmoid. Default: sigmoid", type=str)
    parser.add_argument("--dropout", default = "0.0", help="Dropout probability of first three layers. Default: 0", type=float)
    parser.add_argument("--nepochs", default = 100, help="Number of epochs to run. Default: 100", type=int)
    parser.add_argument("--minibatch", default = 50, help="Mini batch size. Default: 50", type=int)


    print(parser.parse_args())
    run(parser.parse_args())
