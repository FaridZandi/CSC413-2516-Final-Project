import torch
import torch.nn as nn
import torch.optim as optim


def train(train_opts, net, device):
    learning_rate = train_opts["learning_rate"]
    batch_size = train_opts["batch_size"]
    multi_batch_count = train_opts["multi_batch_count"]
    dataloaders = train_opts["dataloaders"]
    dataset_sizes = train_opts["dataset_sizes"]

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(train_opts["epochs"]):  # loop over the dataset multiple times
        logs = {}
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

                if i % multi_batch_count == multi_batch_count - 1:
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

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()

        # liveloss.update(logs)
        # liveloss.send()
        print("epoch:" + str(epoch))
        print(logs)
    print('Finished Training')
