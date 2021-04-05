import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CancerDetectModel(nn.Module):
    def __init__(self):
        super(CancerDetectModel, self).__init__()
        self.name = "CancerDetectModel"
        self.conv1 = torch.nn.Conv2d(3, 32, 5, 5)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 4, 2)
        self.fc = nn.Linear(3 * 3 * 128, 2)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 3 * 3 * 128)
        x = self.fc(x)
        return x

def get_data_loader(batch_size, file_path):
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.ImageFolder(file_path + \
                                                '/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(file_path + \
                                                '/validation', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(file_path + \
                                                  '/test', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, \
                                      batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, \
                                      batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, \
                                      batch_size = batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

filepath = "./breast-histopathology-imagesNEW"
batch_size = 32
train_loader, val_loader, test_loader = get_data_loader(batch_size, filepath)

def train_network(network, train_loader, val_loader, num_epochs = 30, \
                  batch_size = 32, learning_rate = 0.001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), learning_rate)
    
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        for input, label in iter(train_loader):
            output = network(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        model_path = get_model_name(network.name, batch_size, learning_rate, epoch)
        torch.save(network.state_dict(), model_path)
        train_loss[epoch], train_acc[epoch] = accuracy(network, train_loader, \
                                                   criterion)
        val_loss[epoch], val_acc[epoch] = accuracy(network, val_loader, criterion)

        print("Epoch: {}, Training loss: {}, Training accuracy: {}, \
        Validation loss: {}, Validation accuracy: {}".format(
              epoch, 
              train_loss[epoch], 
              train_acc[epoch], 
              val_loss[epoch], 
              val_acc[epoch]))
        
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    
    return model_path, criterion
  
def accuracy(network, loader, criterion):
    correct = 0.
    count = 0.
    loss = 0.
    for input, label in iter(loader):
        output = network(input)
        loss = criterion(output, label)
        result = torch.argmax(output, 1)
        correct += torch.eq(result, label).sum().item()
        loss += loss.item()
        count += len(label)
        
    return (loss / count, correct / count)

def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,batch_size,\
                                                   learning_rate,epoch)
    return path

def plot_training_curves(path):
    train_acc = np.loadtxt("{}_train_acc.csv".format(model_path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(model_path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(model_path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(model_path))
    plt.title("Train vs. Validation Loss")
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label = 'Validation')
    plt.legend(loc = 'best')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Train vs. Validation Acc")
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.legend(loc = 'best')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

best_model = CancerDetectModel()
model_path = "./model_files/model_CancerDetectModel_bs64_lr0.001_epoch2"
state = torch.load(model_path)
best_model.load_state_dict(state)
criterion = nn.CrossEntropyLoss()
loss, accuracy = accuracy(best_model, test_loader, criterion) #final test
print("Test accuracy: {}".format(accuracy))