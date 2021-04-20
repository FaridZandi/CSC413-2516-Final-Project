import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def CIFAR10(batch_size=200, output_size=32):
    transform = transforms.Compose(
        [transforms.Resize(output_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_datasets = {}
    dataloaders = {}

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

    image_datasets['test'] = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                          download=True, transform=transform)

    image_datasets['train'], image_datasets['val'] = torch.utils.data.random_split(train_set, [45000, 5000])

    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                       shuffle=True, num_workers=2)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                      shuffle=False, num_workers=2)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return image_datasets, dataloaders, dataset_sizes, 10
