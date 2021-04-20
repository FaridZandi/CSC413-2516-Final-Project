import torch
from DataLoaders.CIFAR10 import CIFAR10
from DataLoaders.CIFAR100 import CIFAR100
from DataLoaders.TinyImageNet import TinyImageNet

from models.CombModel import CombModel
from models.Resnet_small import resnet18, resnet34, resnet50, resnet101
from models.InceptionV3_small import inception_v3
from models.Stupid import StupidNet
from models.vggnet_small import VGG

from training_loop import train


def make_config():
    config = []
    config += [("Conv1x1", 64)]
    config += [("Basic", 64)] * 1
    config += [("Max", 2)]
    config += [("Conv1x1", 128)]
    config += [("Basic", 128)] * 1
    config += [("Max", 2)]
    config += [("Conv1x1", 256)]
    config += [("Incep", 256)] * 4
    config += [("Max", 2)]
    return config


def main():
    batch_size = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, dataloaders, dataset_sizes, num_classes = CIFAR10(batch_size, 32)

    config = make_config()
    net = CombModel(num_classes=num_classes, config_list=config)
    net = net.to(device)

    train_opts = {
        "epochs": 100,
        "learning_rate": 0.0001,
        "batch_size": batch_size,
        "multi_batch_count": 10,
        "dataloaders": dataloaders,
        "dataset_sizes": dataset_sizes,
        "no_progress_epoch_limit": 50
    }

    test_accuracy, test_loss, logs = train(train_opts, net, device, verbose=True, show_log=True, aux=False)
    print()
    print("test accuracy: {} \ntest_loss: {}".format(test_accuracy, test_loss))

if __name__ == "__main__":
    main()
