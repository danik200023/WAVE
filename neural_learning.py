import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(49 * 25, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 55)

    def forward(self, x):
        """x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class_names = ['A', 'A0', 'B', 'B0', 'D', 'D0', 'DZ', 'DZ0', 'DZH', 'DZH0', 'E0', 'F', 'F0', 'G', 'G0', 'GH', 'I', 'I0',
               'J0', 'K', 'K0', 'KH', 'KH0', 'L', 'L0', 'M', 'M0', 'N', 'N0', 'O', 'O0', 'P', 'P0', 'R', 'R0', 'S',
               'S0', 'SH', 'SH0', 'T', 'T0', 'TS', 'TS0', 'TSH', 'TSH0', 'U', 'U0', 'V', 'V0', 'Y', 'Y0', 'Z', 'Z0',
               'ZH', 'ZH0']
phonemes_label = np.load("phonemes_label.npy")
phonemes_data = np.load("phonemes_data.npy")
phonemes_data = np.nan_to_num(phonemes_data)
X_train, X_test, y_train, y_test = train_test_split(phonemes_data, phonemes_label, test_size=0.25, shuffle=True)
X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_train_torch = torch.from_numpy(y_train).long()
y_test_torch = torch.from_numpy(y_test).long()
train = data_utils.TensorDataset(X_train_torch, y_train_torch)
test = data_utils.TensorDataset(X_test_torch, y_test_torch)
train_loader = data_utils.DataLoader(train, batch_size=400)
test_loader = data_utils.DataLoader(test, batch_size=400)
batch_size = 200
learning_rate = 0.01
epochs = 10
log_interval = 10
net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.NLLLoss()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss))
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    net_out = net(data)
    test_loss += criterion(net_out, target)
    pred = net_out.data.max(1)[1]
    correct += pred.eq(target.data).sum()
test_loss /= len(test_loader.dataset)
print('nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)n'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
