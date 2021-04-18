import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch import Tensor
from Livelossplot import livelossplot

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import warnings

from typing import Callable, Any, Optional, Tuple, List
from collections import namedtuple

from models.vggnet_small import VGG
from models.Resnet_small import resnet18

BATCH_SIZE = 50
IMAGE_SIZE = 32
MULTI_BATCH_COUNT = 100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image_datasets = {}
dataloaders = {}

image_datasets['train'] = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

image_datasets['val'] = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transform)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=2)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = resnet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(net)


def train(train_opts):
    loss_func = nn.CrossEntropyLoss()
    learning_rate = train_opts["learning_rate"]
    liveloss = livelossplot.PlotLosses()
    for epoch in range(train_opts["epochs"]):  # loop over the dataset multiple times
        logs = {}
        for phase in ["train", "val"]:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            optimizer = optim.Adam(net.parameters(), lr=learning_rate)

            epoch_loss = 0.0
            epoch_corrects = 0
            running_loss = 0.0
            running_corrects = 0
            epoch_count = 0

            for i, data in enumerate(dataloaders[phase], 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs)
                loss = loss_func(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # learning_rate = learning_rate * 0.9

                _, preds = torch.max(outputs, 1)
                loss = loss.detach() * inputs.size(0)
                corrects = torch.sum(preds == labels.data)
                epoch_loss += loss
                running_loss += loss
                epoch_corrects += corrects
                running_corrects += corrects
                epoch_count += inputs.size(0)
                # print statistics

                if i % MULTI_BATCH_COUNT == MULTI_BATCH_COUNT - 1:
                    print("[{}/{}] {}: running_loss: {}, running_corrects:{}".format(
                        (i + 1) * BATCH_SIZE,
                        dataset_sizes[phase],
                        phase,
                        running_loss / MULTI_BATCH_COUNT / BATCH_SIZE,
                        running_corrects / MULTI_BATCH_COUNT / BATCH_SIZE))

                    running_loss = 0.0
                    running_corrects = 0

            epoch_loss = epoch_loss / epoch_count
            epoch_acc = epoch_corrects.float() / epoch_count

            prefix = ''
            if phase == 'val':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.send()
    print('Finished Training')


train_opts = {
    "epochs": 50,
    "learning_rate": 0.001
}

train(train_opts)