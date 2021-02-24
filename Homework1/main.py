
"""
Assignment 1
Class: IFT6135 - Representation learning
Author: Eric Pfleiderer
"""
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from Homework1.src.solution import NN, load_cifar10
from Homework1.src.models import NN2, CNN

number_epochs = 20

# Flat/2d CIFAR datasets
flat_train_data, flat_valid_data, flat_test_data = load_cifar10('data/', flatten=True)
flat_data = (flat_train_data, flat_valid_data, flat_test_data)
train_data, valid_data, test_data = load_cifar10('data/', flatten=False)
data = (train_data, valid_data, test_data)


def compare_inits():

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
def validate_config(l, a):
    neural_net = NN(hidden_dims=l,
                    epsilon=1e-6,
                    lr=0.01,
                    batch_size=64,
                    seed=1,
                    activation=a,
                    data=flat_data)

    neural_net.train_loop(number_epochs)
    final_val = neural_net.train_logs['validation_accuracy'][-1]

    with open('logs/validation_results.txt', 'a') as f:
        f.write(f'Final validation accuracy for shape:{l} and activation:{a} is {final_val}.\n')

    print(f'Final validation accuracy for shape:{l} and activation:{a} is {final_val}.\n')


def launch_grid_search():
    hparam_space = {'layers': [(764, 256),
                               (1024, 512, 64, 64)],
                    'activation': ['relu', 'tanh', 'sigmoid']
                    }

    processes = []

    with open('logs/validation_results.txt', 'a') as f:
        f.write('----------------------GRID SEARCH START----------------------\n')
        f.write(f'Training for {number_epochs} epochs.\n')

    # Spin up parallel processes to speed things up
    for layers in hparam_space['layers']:
        for activation in hparam_space['activation']:
            validate_config(layers, activation)

            process = Process(target=validate_config, args=(layers, activation))
            processes.append(process)
            process.start()

    for p in processes:
        p.join()

    with open('logs/validation_results.txt', 'a') as f:
        f.write(f'----------------------GRID SEARCH END----------------------\n')


# Q4
def Q4():
    pass


def train_CNN():
    net = CNN()
    # device = torch.device('cuda')
    # net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    train_x, train_y = train_data[0], train_data[1]
    train_y = np.argmax(train_y, axis=1)

    tensor_x, tensor_y = torch.Tensor(train_x), torch.Tensor(train_y)
    train_loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=64, shuffle=True, num_workers=1)
    # val_loader = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=1)

    training_loss = []

    for epoch in range(number_epochs):
        print('epoch: ', epoch)
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())


if __name__ == '__main__':
    # Q2
    compare_inits()

    # Q3()
    launch_grid_search()

    # Q4
    # TODO: code it

    # Q5
    train_CNN()
