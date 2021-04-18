import torch
from DataLoaders.CIFAR100 import CIFAR100
from models.Resnet_small import resnet18
from training_loop import train

batch_size = 50

image_datasets, dataloaders, dataset_sizes, num_classes = CIFAR100(batch_size)

net = resnet18(num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

train_opts = {
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "multi_batch_count": 100,
    "dataloaders": dataloaders,
    "dataset_sizes": dataset_sizes,
}

train(train_opts, net, device)