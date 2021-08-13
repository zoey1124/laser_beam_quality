from psana import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import cv2

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from torchvision import models
import time

#############
#  DataSet  #
#############

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir is the directory of HDF5 file.
        """
        self.f = h5py.File(img_dir, "r")
        self.img = self.f['image']
        self.label = self.f['label']
        self.transform = transform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        image = self.img[idx]
        if self.transform:
            image = self.transform(image)
        label = self.label[idx]
        return image, label

def get_dataloader(dset, validation_split, batch_size=16, random_seed=0):
    """
    dset: CustomDataset type
    batch_size: int
    validation_split: float, represent percentage of dataset used as validation
    """
    dataset_size = len(dset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
  
    train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, sampler=valid_sampler, num_workers=4)
    return train_loader, valid_loader

###########
#  Model  #
###########
class CNN_128(nn.Module):
    """
    The image in dataset should be 128 * 128.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*31*31, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=2)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNN_28_1(nn.Module):
    """
    The input image should be 28 * 28.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CNN_28_2(nn.Module):
    """
    The input image should be 28 * 28.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


###########
#  Train  #
###########
def train(model_name, model, train_loader, valid_loader, num_epochs = 1):
    """
    model_name: string type. Must choose from 1.CNN_128 2.CNN_28_1 3.CNN_28_2
    num_epochs: int type. The total number of training epochs 
    """
    optim = torch.optim.SGD(model.parameters(), lr = 0.001)
    criterian = nn.CrossEntropyLoss()
    
    train_loss_lst = []
    val_loss_lst = []
    train_acc_lst = []
    val_acc_lst = []
    
    start_time = time.time()
    loss_min = np.inf
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train_total, train_correct = 0, 0
        loss_train, loss_valid = 0, 0
        
        model.train()
        for step in range(1, len(train_loader) + 1): # number of batches
            optim.zero_grad()
            inputs, targets = next(iter(train_loader))
            targets = targets.type(torch.LongTensor)
            outputs = model(inputs)
            loss_train_step = criterian(outputs, targets)
            loss_train_step.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            loss_train += loss_train_step.item()
            
        val_total, val_correct = 0, 0
        
        model.eval()
        with torch.no_grad():
            for step in range(1, len(valid_loader) + 1):
                inputs, targets = next(iter(train_loader))
                targets = targets.type(torch.LongTensor)
                outputs = model(inputs)
                loss_valid_step = criterian(outputs, targets)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                loss_valid += loss_valid_step.item()
            
        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        
        train_loss_lst.append(loss_train)
        val_loss_lst.append(loss_valid)
        train_acc_lst.append(train_correct / train_total)
        val_acc_lst.append(val_correct / val_total)
        
        print("Epoch: {}; Train Loss: {}; Train Accuracy: {}; Valid Loss:{}; Valid Accuracy: {}"
              .format(epoch, loss_train, train_correct / train_total, loss_valid, val_correct / val_total))
        
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(model.state_dict(), f'./models/{model_name}.pth')
        
        print("Epoch elapsed time: {}".format(time.time() - epoch_start_time))
        print("-------------------------------")
        
    with open(f'./loss/{model_name}_train_loss.npy', 'wb') as f:
        np.save(f, np.array(train_loss_lst))
    with open(f'./loss/{model_name}_valid_loss.npy', 'wb') as f:
        np.save(f, np.array(val_loss_lst))
    with open(f'./accuracy/{model_name}_train_accuracy.npy', 'wb') as f:
        np.save(f, np.array(train_acc_lst))
    with open(f'./accuracy/{model_name}_val_accuracy.npy', 'wb') as f:
        np.save(f, np.array(val_acc_lst))
        
    print("Train complete.")
    print("Total Elapsed Time: {}".format(time.time() - start_time))


#################
#  Performance  #
#################
def compare_loss(files):
    """
    files is a list of file names in loss directory
    """
    for file in files:
        lst = np.load('./loss/' + file)
        plt.plot(np.arange(1, len(lst) + 1), list(lst), label=file.strip('.npy'))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def compare_acc(files):
    """
    files are a list of file names in accuracy directory
    """
    for file in files:
        lst = np.load('./accuracy/' + file)
        plt.plot(np.arange(1, len(lst) + 1), list(lst), label=file.strip('.npy'))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def load_model(model_type):
    model = None
    if model_type == 'CNN_128':
        model = CNN_128()
    elif model_type == 'CNN_28_1':
        model = CNN_28_1()
    elif model_type == 'CNN_28_2':
        model = CNN_28_2()
    else:
        raise Exception("model type not recognized.")
    model.load_state_dict(torch.load(model_type + '.pth'))
    return model

