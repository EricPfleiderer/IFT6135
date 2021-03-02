
"""
Assignment 1
Class: IFT6135 - Representation learning
Author: Eric Pfleiderer
"""

import logging
import sys
import time
from multiprocessing import Process
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from Homework1.src.solution import NN, load_cifar10
from Homework1.src.models import NN2, CNN

# Flat/2d CIFAR datasets
flat_train_data, flat_valid_data, flat_test_data = load_cifar10('data/', flatten=True)
flat_data = (flat_train_data, flat_valid_data, flat_test_data)
train_data, valid_data, test_data = load_cifar10('data/', flatten=False)
data = (train_data, valid_data, test_data)


def compare_inits(number_epochs=20):

    glorot_net = NN2(hidden_dims=(784, 256),
                     epsilon=1e-6,
                     lr=0.01,
                     batch_size=64,
                     seed=1,
                     activation="relu",
                     data=flat_data)

    norm_net = NN2(hidden_dims=(784, 256),
                   epsilon=1e-6,
                   lr=0.01,
                   batch_size=64,
                   seed=1,
                   activation="relu",
                   data=flat_data)

    print('Training glorot net...')
    glorot_net.train_loop(number_epochs, init_type='glorot')

    print('Training norm net...')
    norm_net.train_loop(number_epochs, init_type='normal')

    x = range(number_epochs)
    y_glorot = glorot_net.train_logs['train_loss']
    y_normal = norm_net.train_logs['train_loss']

    plt.figure()
    plt.plot(x, y_glorot, label='glorot')
    plt.plot(x, y_normal, label='normal')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.legend(loc='best')
    plt.savefig('imgs/q2_train_loss_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')


# Train a single model from config and write validation results to file
def validate_config(l, a, number_epochs=20):
    neural_net = NN(hidden_dims=l,
                    epsilon=1e-6,
                    lr=0.01,
                    batch_size=64,
                    seed=1,
                    activation=a,
                    data=flat_data)

    neural_net.train_loop(number_epochs)
    final_val = neural_net.train_logs['validation_accuracy'][-1]

    logging.info(f'Final validation accuracy for shape:{l} and activation:{a} is {final_val}.\n')


def launch_grid_search(number_epochs=20):
    hparam_space = {'layers': [(764, 256),
                               (1024, 512, 64, 64)],
                    'activation': ['relu', 'tanh', 'sigmoid']
                    }

    logging.info('----------------------GRID SEARCH START----------------------\n')
    logging.info(f'Training for {number_epochs} epochs.\n')

    # Spin up parallel processes to speed things up
    processes = []
    for layers in hparam_space['layers']:
        for activation in hparam_space['activation']:
            validate_config(layers, activation)

            process = Process(target=validate_config, args=(layers, activation))
            processes.append(process)
            process.start()

    for p in processes:
        p.join()

    logging.info(f'----------------------GRID SEARCH END----------------------\n')


def train_CNN(number_epochs=20, dropout=False, L2=False, batch_norm=False, first_pass=False):

    # Split data by features / targets
    train_x, train_y = data[0][0], data[0][1]
    val_x, val_y = data[1][0], data[1][1]

    # Collapse onehot labels
    train_y, val_y = np.argmax(train_y, axis=1), np.argmax(val_y, axis=1)

    # Convert data to torch tensors and wrap in a dataloader
    tensor_x_train, tensor_y_train = torch.Tensor(train_x), torch.Tensor(train_y)
    train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=32, shuffle=True)
    tensor_x_val, tensor_y_val = torch.Tensor(val_x), torch.Tensor(val_y)
    val_loader = DataLoader(TensorDataset(tensor_x_val, tensor_y_val), batch_size=256, shuffle=False)

    # Train the model
    net = CNN(train_loader, val_loader, optim.SGD, nn.CrossEntropyLoss, dropout, L2=L2, batch_norm=batch_norm, first_pass=first_pass)

    for epoch in range(number_epochs):
        logging.info('-----------------------------------')
        logging.info('epoch: ' + str(epoch))
        net.step()

        for k, v in net.train_logs.items():
            logging.info(k + ": " + str(round(v[-1], 4)))

        logging.info('-----------------------------------')

    return net


def compare_gradients(number_epochs=20):

    net = NN(hidden_dims=(1024, 512, 64, 64),
             epsilon=1e-6,
             lr=0.01,
             batch_size=64,
             seed=1,
             activation="relu",
             data=flat_data)

    print('Training glorot net...')
    net.train_loop(number_epochs)

    # Single data point (batched)
    x = flat_train_data[0][5][None, :]
    target = flat_train_data[1][5][None, :]

    # Get 100 first parameters of the last layer
    layer_id = net.n_hidden+1
    weights = net.weights[f'W{layer_id}']
    N = np.array([10**i for i in range(1, 6)])
    p = np.minimum(100, weights.size)

    # Compute finite differences
    finite_diffs = np.empty(shape=(N.size, p))
    for i, n in enumerate(N):
        epsilon = 1/n
        for j in range(p):
            idx = j // weights.shape[1]
            idy = j % weights.shape[1]

            # Delta -
            weights[idx, idy] -= epsilon
            pred_m = net.forward(x)[f'Z{layer_id}']
            loss_m = net.loss(pred_m, target)
            weights[idx, idy] += epsilon

            # Delta +
            weights[idx, idy] += epsilon
            pred_p = net.forward(x)[f'Z{layer_id}']
            loss_p = net.loss(pred_p, target)
            weights[idx, idy] -= epsilon

            finite_diffs[i, j] = (loss_p-loss_m)/(2*epsilon)

    # Compute exact derivatives
    forward = net.forward(x)
    grads = net.backward(forward, target)[f'dW{layer_id}']
    exact_diffs = np.empty(shape=(100,))
    for m in range(p):
        idx = m // grads.shape[1]
        idy = m % grads.shape[1]
        exact_diffs[m] = grads[idx, idy]

    # Compute difference between derivatives
    diffs = np.abs(finite_diffs - exact_diffs)  # 5x100 and 100
    max_diffs = np.max(diffs, axis=1)

    plt.figure()
    plt.plot(N, max_diffs)
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel('max diff')
    plt.savefig('imgs/maxdiff_vs_n_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')


def plot_cnn_vs_nn(number_epochs=20, first_pass=False):

    cnn = train_CNN(number_epochs=number_epochs, first_pass=first_pass)

    nn = NN2(hidden_dims=(1024, 512, 64, 64),
             epsilon=1e-6,
             lr=0.01,
             batch_size=64,
             seed=1,
             activation="relu",
             data=flat_data)

    print('Training glorot net...')
    nn.train_loop(number_epochs, first_pass=first_pass)

    plt.figure()
    plt.plot(range(len(cnn.train_logs['train_loss'])), cnn.train_logs['train_loss'], label='cnn train_loss', linestyle="--")
    plt.plot(range(len(cnn.train_logs['validation_loss'])), cnn.train_logs['validation_loss'], label='cnn validation_loss', linestyle="--")
    plt.plot(range(len(nn.train_logs['train_loss'])), nn.train_logs['train_loss'], label='nn train_loss')
    plt.plot(range(len(nn.train_logs['validation_loss'])), nn.train_logs['validation_loss'], label='nn validation_loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('imgs/cnn_vs_nn_loss_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')

    plt.figure()
    plt.plot(range(len(cnn.train_logs['train_accuracy'])), cnn.train_logs['train_accuracy'], label='cnn train_accuracy', linestyle="--")
    plt.plot(range(len(cnn.train_logs['validation_accuracy'])), cnn.train_logs['validation_accuracy'], label='cnn validation_accuracy', linestyle="--")
    plt.plot(range(len(nn.train_logs['train_accuracy'])), nn.train_logs['train_accuracy'], label='nn train_accuracy')
    plt.plot(range(len(nn.train_logs['validation_accuracy'])), nn.train_logs['validation_accuracy'], label='nn validation_accuracy')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('imgs/cnn_vs_nn_accuracy_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')


def compare_regularization(number_epochs=20, dropout=False, L2=False, batch_norm=False):

    vanilla_net = train_CNN(number_epochs)
    reg_net = train_CNN(number_epochs, dropout=dropout, L2=L2, batch_norm=batch_norm)

    plt.figure()
    plt.plot(range(number_epochs), vanilla_net.train_logs['train_loss'], label='vanilla train_loss', linestyle="--")
    plt.plot(range(number_epochs), vanilla_net.train_logs['validation_loss'], label='vanilla validation_loss', linestyle="--")
    plt.plot(range(number_epochs), reg_net.train_logs['train_loss'], label='regularized train_loss')
    plt.plot(range(number_epochs), reg_net.train_logs['validation_loss'], label='regularized validation_loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('imgs/vanilla_vs_reg_loss_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')

    plt.figure()
    plt.plot(range(number_epochs), vanilla_net.train_logs['train_accuracy'], label='vanilla train_accuracy', linestyle="--")
    plt.plot(range(number_epochs), vanilla_net.train_logs['validation_accuracy'], label='vanilla validation_accuracy', linestyle="--")
    plt.plot(range(number_epochs), reg_net.train_logs['train_accuracy'], label='regularized train_accuracy')
    plt.plot(range(number_epochs), reg_net.train_logs['validation_accuracy'], label='regularized validation_accuracy')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('imgs/vanilla_vs_reg_accuracy_' + time.strftime("%Y%m%d-%H%M%S") + '.pdf')


if __name__ == '__main__':

    # Add logger
    logging_level = logging.INFO
    logging.basicConfig(filename='logs/general.txt')
    root = logging.getLogger()
    root.setLevel(logging_level)

    # Add handler to std out
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    root.addHandler(handler)

    # Q2
    # compare_inits()

    # Q3()
    # launch_grid_search()

    # Q4
    # compare_gradients(20)

    # Q5
    # plot_cnn_vs_nn(20)
    compare_regularization(80, batch_norm=True)
