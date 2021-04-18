import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 50
IMAGE_SIZE = 32

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

