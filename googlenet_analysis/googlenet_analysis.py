'''
GoogleNet Analysis

GoogLeNet:
    https://arxiv.org/pdf/1409.4842v1.pdf
    https://pytorch.org/hub/pytorch_vision_googlenet/
    https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py
CIFAR:
    https://pytorch.org/vision/stable/datasets.html#cifar
    https://www.cs.toronto.edu/~kriz/cifar.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(549)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data():
    def __init__(self, opts):
        self.dataset = opts["dataset"]
        self.data_path = opts["data_path"]
        self.download = opts["download"]
        self.batch_size = opts["batch_size"]
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def train_loader(self):
        if self.dataset == "CIFAR10":
            data = datasets.CIFAR10(r"{}\CIFAR10".format(self.data_path), download=self.download, transform=self.preprocess)
        elif self.dataset == "CIFAR100":
            data = datasets.CIFAR100(r"{}\CIFAR100".format(self.data_path), download=self.download, transform=self.preprocess)
        else:
            raise Exception("dataset must be CIFAR10 or CIFAR100")
        train_data, val_data = random_split(data, [int(len(data)*0.85), int(len(data)*0.15)])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_loader, val_loader
    def test_loader(self):
        if self.dataset == "CIFAR10":
            data = datasets.CIFAR10(r"{}\CIFAR10".format(self.data_path), download=self.download, transform=self.preprocess, train=False)
        elif self.dataset == "CIFAR100":
            data = datasets.CIFAR100(r"{}\CIFAR100".format(self.data_path), download=self.download, transform=self.preprocess, train=False)
        else:
            raise Exception("dataset must be CIFAR10 or CIFAR100")
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return dataloader

def train(model, train_loader, val_loader, opts):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opts["learning_rate"], momentum=opts["momentum"])
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: opts["lr_decay"], verbose=True)
    num_iter = 0
    iters, losses, train_acc, val_acc = [], [], [], []
    for epoch in range(1, opts["num_epochs"] + 1):
        if epoch % 8 == 0:
            scheduler.step()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if opts["aux_logits"]:
                loss_out = criterion(outputs.logits, labels)
                loss_aux1 = criterion(outputs.aux_logits1, labels)
                loss_aux2 = criterion(outputs.aux_logits2, labels)
                loss = loss_out + 0.3 * (loss_aux1 + loss_aux2)
            else:
                loss = criterion(outputs, labels)
            loss /= opts["batch_size"]
            loss.backward()
            optimizer.step()
            num_iter += 1
        iters.append(num_iter)
        losses.append(loss.item())
        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, val_loader))
        print("Epoch: {:2} | Loss: {:.4f} | Train acc: {:.4f} | Val acc: {:.4f}".format(epoch, loss.item(), train_acc[-1], val_acc[-1]))
    plot_training_curve(iters, losses, train_acc, val_acc, opts)
    return model

def get_accuracy(model, dataloader):
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(inputs).logits
        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def plot_training_curve(iters, losses, train_acc, val_acc, opts):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("saved_files/images/{}_{}_bs{}_ep{}_lr{}.png".format(
        opts["model"], opts["dataset"], opts["batch_size"], opts["num_epochs"], opts["learning_rate"]))

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class StrippedGoogLeNet(models.GoogLeNet):
    # googlenet w/o inception layers, instead basic conv layers
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inception3a = BasicConv2d(192, 256, kernel_size=1, stride=1, padding=0)
        self.inception3b = BasicConv2d(256, 480, kernel_size=1, stride=1, padding=0)
        self.inception4a = BasicConv2d(480, 512, kernel_size=1, stride=1, padding=0)
        self.inception4b = BasicConv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.inception4c = BasicConv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.inception4d = BasicConv2d(512, 528, kernel_size=1, stride=1, padding=0)
        self.inception4e = BasicConv2d(528, 832, kernel_size=1, stride=1, padding=0)
        self.inception5a = BasicConv2d(832, 832, kernel_size=1, stride=1, padding=0)
        self.inception5b = BasicConv2d(832, 1024, kernel_size=1, stride=1, padding=0)

def create_model(opts):
    if opts["dataset"] == "CIFAR10":
        num_classes = 10
    elif opts["dataset"] == "CIFAR100":
        num_classes = 100
    else:
        raise Exception("dataset must be CIFAR10 or CIFAR100")
    if opts["model"] == "googlenet":
        model = models.googlenet(num_classes=num_classes, aux_logits=opts["aux_logits"])
    elif opts["model"] == "stripped_googlenet":
        model = StrippedGoogLeNet(num_classes=num_classes, aux_logits=opts["aux_logits"])
    else:
        raise Exception("model must be googlenet or stripped_googlenet")
    return model

if __name__ == "__main__":
    opts = {
        "model": "stripped_googlenet",
        "data_path": r"H:\Saad\Programming\Datasets",
        "download": False,
        "dataset": "CIFAR10",
        "train_model": False,
        "aux_logits": True,
        "batch_size": 64,
        "num_epochs": 30,
        "learning_rate": 0.01,
        "lr_decay": 0.96,
        "momentum": 0.9
    }

    data = Data(opts)
    train_loader, val_loader = data.train_loader()
    test_loader = data.test_loader()

    model = create_model(opts)
    model.to(device)

    if opts["train_model"]:
        model = train(model, train_loader, val_loader, opts)
        torch.save(model, "saved_files/models/{}_{}_bs{}_ep{}_lr{}.pth".format(
            opts["model"], opts["dataset"], opts["batch_size"], opts["num_epochs"], opts["learning_rate"]))
    else:
        model = torch.load("saved_files/models/{}_{}_bs{}_ep{}_lr{}.pth".format(
            opts["model"], opts["dataset"], opts["batch_size"], opts["num_epochs"], opts["learning_rate"]))
        print("Test accuracy: {:.4f}".format(get_accuracy(model, test_loader)))
