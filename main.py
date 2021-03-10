from lstm_solution import LSTM

from utils.wikitext2 import Wikitext2
from torch.utils.data import DataLoader
import torch


train_data = Wikitext2(root='data/', split='train')
validation_data = Wikitext2(root='data/', split='validation')
test_data = Wikitext2(root='data/', split='test')

train_loader = DataLoader(train_data, batch_size=16)

net = LSTM()
device = torch.device('cuda')
net.to(device)

for i, data in enumerate(train_loader):
    print(i)
    x, (h, c) = net(data['source'].to(device), net.initial_states(data['source'].shape[0]))
    loss = net.loss(x, data['target'].to(device), data['mask'].to(device))

    # x, (h, c) = net(data['source'], net.initial_states(data['source'].shape[0]))
    # loss = net.loss(x, data['target'], data['mask'])
