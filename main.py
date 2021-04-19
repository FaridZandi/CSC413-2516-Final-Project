import torch
from DataLoaders.CIFAR10 import CIFAR10
from DataLoaders.CIFAR100 import CIFAR100
from DataLoaders.TinyImageNet import TinyImageNet

from models.Resnet_small import resnet18, resnet34, resnet50, resnet101
from models.InceptionV3 import inception_v3
from models.Stupid import StupidNet
from models.vggnet_small import VGG

from training_loop import train


def main():
    batch_size = 50

    image_datasets, dataloaders, dataset_sizes, num_classes = CIFAR10(batch_size)

    net = resnet101(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(net)

    train_opts = {
        "epochs": 100,
        "learning_rate": 0.0001,
        "batch_size": batch_size,
        "multi_batch_count": 100,
        "dataloaders": dataloaders,
        "dataset_sizes": dataset_sizes,
    }

    best_val_accuracy, best_val_loss, logs = train(train_opts, net, device)

    print(best_val_accuracy, best_val_loss)


if __name__ == "__main__":
    main()
