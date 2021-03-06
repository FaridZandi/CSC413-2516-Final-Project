import string
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim


def train(train_opts, net, device, verbose=True, show_log=True, aux=False):
    letters = string.ascii_lowercase
    PATH = './nets/net-' + ''.join(random.choice(letters) for i in range(10)) + '.pth'

    learning_rate = train_opts["learning_rate"]
    batch_size = train_opts["batch_size"]
    multi_batch_count = train_opts["multi_batch_count"]
    dataloaders = train_opts["dataloaders"]
    dataset_sizes = train_opts["dataset_sizes"]

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_val_loss = 100000000000
    best_val_accuracy = 0
    logs = []

    no_progress_epochs = 0
    no_progress_epoch_limit = train_opts["no_progress_epoch_limit"]


    for epoch in range(train_opts["epochs"]):  # loop over the dataset multiple times
        log = {}
        epoch_start = time.time()

        for phase in ["train", "val"]:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            running_loss = 0.0
            running_corrects = 0
            epoch_count = 0

            for i, data in enumerate(dataloaders[phase], 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                if phase == "val":
                    with torch.no_grad():
                        if aux:
                            outputs, _ = net(inputs)
                        else:
                            outputs = net(inputs)
                        loss = loss_func(outputs, labels)
                else:
                    if aux:
                        outputs, _ = net(inputs)
                    else:
                        outputs = net(inputs)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                loss = loss.detach() * inputs.size(0)
                corrects = torch.sum(preds == labels.data)
                epoch_loss += loss
                running_loss += loss
                epoch_corrects += corrects
                running_corrects += corrects
                epoch_count += inputs.size(0)


                if i % multi_batch_count == multi_batch_count - 1:
                    if verbose:
                        print("[{}/{}] {}: running_loss: {}, running_corrects:{}".format(
                            (i + 1) * batch_size,
                            dataset_sizes[phase],
                            phase,
                            running_loss / multi_batch_count / batch_size,
                            running_corrects / multi_batch_count / batch_size))

                    running_loss = 0.0
                    running_corrects = 0

            epoch_loss = epoch_loss / epoch_count
            epoch_acc = epoch_corrects.float() / epoch_count

            prefix = ''
            if phase == 'val':
                prefix = 'val_'

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_accuracy = epoch_acc
                    no_progress_epochs = 0
                    torch.save(net.state_dict(), PATH)

                else:
                    no_progress_epochs += 1
                    if verbose:
                        print("no_progress_epochs: ", no_progress_epochs)

            log[prefix + 'loss'] = epoch_loss.item()
            log[prefix + 'accuracy'] = epoch_acc.item()

        if verbose:
            print("epoch: {} finished in {} seconds".format(epoch, time.time() - epoch_start))

        if show_log:
            print(log)

        logs.append(log)

        if no_progress_epochs > no_progress_epoch_limit:
            break

    if verbose:
        print("best val accuracy", best_val_accuracy.item())
        print("best val loss", best_val_loss.item())


    net.load_state_dict(torch.load(PATH))
    net.eval()

    test_loss = 0
    test_corrects = 0
    test_count = 0

    for i, data in enumerate(dataloaders["test"], 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            if aux:
                outputs, _ = net(inputs)
            else:
                outputs = net(inputs)
            loss = loss_func(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss = loss.detach() * inputs.size(0)
        corrects = torch.sum(preds == labels.data)
        test_loss += loss
        test_corrects += corrects
        test_count += inputs.size(0)

    test_loss = test_loss / test_count
    test_accuracy = test_corrects.float() / test_count

    return test_accuracy.item(), test_loss.item(), logs
