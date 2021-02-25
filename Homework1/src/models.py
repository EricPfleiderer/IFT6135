import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from Homework1.src.solution import NN


# Redefine neural network to include normal initialiazation
class NN2(NN):

    def initialize_weights(self, dims, init_type='glorot'):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):

            if init_type == 'glorot':
                d = np.sqrt(6 / (all_dims[layer_n - 1] + all_dims[layer_n]))
                self.weights[f"W{layer_n}"] = np.random.uniform(-d, d, size=(all_dims[layer_n - 1], all_dims[layer_n]))

            elif init_type == 'normal':
                self.weights[f"W{layer_n}"] = np.random.normal(0, 0.1, size=(all_dims[layer_n - 1], all_dims[layer_n]))

            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def train_loop(self, n_epochs, init_type='glorot'):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims, init_type=init_type)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            print('epoch: ', epoch)
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                forward = self.forward(minibatchX)
                grads = self.backward(forward, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs


# Q5
class CNN(nn.Module):
    def __init__(self, train_loader, val_loader, optimizer, criterion):
        super(CNN, self).__init__()
        self.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax(dim=1)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss / optimizer
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

        # Training logs
        self.train_logs = dict()
        self.train_logs['train_accuracy'] = []
        self.train_logs['validation_accuracy'] = []
        self.train_logs['train_loss'] = []
        self.train_logs['validation_loss'] = []

    # Forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    # Full training step, including validation
    def step(self):
        self.train_step()
        self.validation_step()

    def train_step(self):

        running_loss = 0.0
        running_hits = 0

        for i, data in enumerate(self.train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels.long())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_hits += np.where(np.argmax(outputs.detach().numpy(), axis=1) == labels.numpy())[0].size

        # Average loss and accuracy over data set for the current epoch
        loss = running_loss / (len(self.train_loader)*self.train_loader.batch_size)
        accuracy = running_hits / (len(self.train_loader) * self.train_loader.batch_size)

        self.train_logs['train_loss'].append(loss)
        self.train_logs['train_accuracy'].append(accuracy)

    @torch.no_grad()
    def validation_step(self):
        running_loss = 0.0
        running_hits = 0

        for i, data in enumerate(self.val_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels.long())

            running_loss += loss.item()
            running_hits += np.where(np.argmax(outputs.numpy(), axis=1) == labels.numpy())[0].size

        # Average loss and accuracy over data set for the current epoch
        loss = running_loss / (len(self.val_loader)*self.val_loader.batch_size)
        accuracy = running_hits / (len(self.val_loader) * self.val_loader.batch_size)

        self.train_logs['validation_loss'].append(loss)
        self.train_logs['validation_accuracy'].append(accuracy)

