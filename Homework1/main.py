
"""
Assignment 1
Class: IFT6135 - Representation learning
Author: Eric Pfleiderer
"""
from multiprocessing import Process
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
    plt.savefig('imgs/q2_train_loss.pdf')


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

    logging.INFO(f'Final validation accuracy for shape:{l} and activation:{a} is {final_val}.\n')


def launch_grid_search(number_epochs=20):
    hparam_space = {'layers': [(764, 256),
                               (1024, 512, 64, 64)],
                    'activation': ['relu', 'tanh', 'sigmoid']
                    }

    logging.INFO('----------------------GRID SEARCH START----------------------\n')
    logging.INFO(f'Training for {number_epochs} epochs.\n')

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

    logging.INFO(f'----------------------GRID SEARCH END----------------------\n')


# Q4
def Q4():
    pass


def train_CNN(number_epochs=20):

    # Split data by features / targets
    train_x, train_y = data[0][0], data[0][1]
    val_x, val_y = data[1][0], data[1][1]

    # Collapse onehot labels
    train_y, val_y = np.argmax(train_y, axis=1), np.argmax(val_y, axis=1)

    # Convert data to torch tensors and wrap in a dataloader
    tensor_x_train, tensor_y_train = torch.Tensor(train_x), torch.Tensor(train_y)
    train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=16, shuffle=True)
    tensor_x_val, tensor_y_val = torch.Tensor(val_x), torch.Tensor(val_y)
    val_loader = DataLoader(TensorDataset(tensor_x_val, tensor_y_val), batch_size=64, shuffle=False)

    # Train the model
    net = CNN(train_loader, val_loader, optim.Adam, nn.CrossEntropyLoss)

    for epoch in range(number_epochs):
        logging.info('-----------------------------------')
        logging.info('epoch: ' + str(epoch))
        net.step()
        logging.info('train_loss:' + str(round(net.train_logs['train_loss'][-1], 4)))
        logging.info('train_accuracy:' + str(round(net.train_logs['train_accuracy'][-1], 2)))
        logging.info('validation_loss:' + str(round(net.train_logs['validation_loss'][-1], 4)))
        logging.info('validation_accuracy:' + str(round(net.train_logs['validation_accuracy'][-1], 2)))
        logging.info('-----------------------------------')

    return net


def plot_cnn_vs_nn(number_epochs=20):

    cnn = train_CNN(number_epochs=number_epochs)

    nn = NN(hidden_dims=(784, 256),
            epsilon=1e-6,
            lr=0.01,
            batch_size=64,
            seed=1,
            activation="relu",
            data=flat_data)

    print('Training glorot net...')
    nn.train_loop(number_epochs)

    plt.figure()
    plt.plot(range(number_epochs), cnn.train_logs['train_loss'], label='cnn train_loss', linestyle="--")
    plt.plot(range(number_epochs), cnn.train_logs['validation_loss'], label='cnn validation_loss', linestyle="--")
    plt.plot(range(number_epochs), nn.train_logs['train_loss'], label='nn train_loss')
    plt.plot(range(number_epochs), nn.train_logs['validation_loss'], label='nn validation_loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(range(number_epochs), cnn.train_logs['train_accuracy'], label='cnn train_accuracy', linestyle="--")
    plt.plot(range(number_epochs), cnn.train_logs['validation_accuracy'], label='cnn validation_accuracy', linestyle="--")
    plt.plot(range(number_epochs), nn.train_logs['train_accuracy'], label='nn train_accuracy')
    plt.plot(range(number_epochs), nn.train_logs['validation_accuracy'], label='nn validation_accuracy')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':

    import sys

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
    # TODO: code it

    # Q5
    plot_cnn_vs_nn(20)
