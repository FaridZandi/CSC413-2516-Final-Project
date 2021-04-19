import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def CIFAR100(batch_size=50, output_size=32):
    transform = transforms.Compose(
        [transforms.Resize(output_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_datasets = {}
    dataloaders = {}

    image_datasets['train'] = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                           download=True, transform=transform)
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                       shuffle=True, num_workers=2)

    image_datasets['val'] = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                         download=True, transform=transform)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                     shuffle=False, num_workers=2)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return image_datasets, dataloaders, dataset_sizes, 100
