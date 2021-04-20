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
    config += [("Basic", 64)] * 2
    config += [("Max", 2)]
    config += [("Conv1x1", 128)]
    config += [("Basic", 128)] * 2
    config += [("Max", 2)]
    config += [("Conv1x1", 256)]
    config += [("Incep", 256)] * 2
    config += [("Max", 2)]
    return config


def main():
    batch_size = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, dataloaders, dataset_sizes, num_classes = CIFAR10(batch_size, 32)

    results = {}
    for type in ["Incep", "IncepRes", "Basic", "Res"]:
        results[type] = {}
        for count in [1, 2, 3]:
            results[type][count] = {}
            config = []
            config += [("Conv1x1", 64)]
            config += [(type, 64)] * count
            config += [("Max", 2)]
            config += [("Conv1x1", 128)]
            config += [(type, 128)] * count
            config += [("Max", 2)]
            config += [("Conv1x1", 256)]
            config += [(type, 256)] * count

            net = CombModel(num_classes=num_classes, config_list=config)
            net = net.to(device)

            train_opts = {
                "epochs": 100,
                "learning_rate": 0.0001,
                "batch_size": batch_size,
                "multi_batch_count": 10,
                "dataloaders": dataloaders,
                "dataset_sizes": dataset_sizes,
                "no_progress_epoch_limit": 5
            }

            test_accuracy, test_loss, logs = train(train_opts, net, device, verbose=False, show_log=True, aux=False)
            results[type][count] = {"test-accuracy": test_accuracy,
                                    "test-loss": test_loss}

            print()
            print("type: {}, count: {}, test accuracy: {} \ntest_loss: {}".format(
                type, count, test_accuracy, test_loss))

    print(results)


if __name__ == "__main__":
    main()

hello = {'Incep': {1: {'test-accuracy': 0.48429998755455017, 'test-loss': 1.9699218273162842},
                   2: {'test-accuracy': 0.38979998230934143, 'test-loss': 2.381167411804199},
                   3: {'test-acc uracy': 0.3393999934196472, 'test-loss': 2.6849215030670166}},
         'IncepRes': {1: {'test-accuracy': 0.49289998412132263, 'test-loss': 1.897254228591919},
                      2: {'test-accuracy': 0.42659997940063477, 'test-loss': 2.206602096557617},
                      3: {'test-accuracy': 0.4032000005245209, 'test-loss': 2.319490671157837}},
         'Basic': {1: {'test-accuracy': 0.505899965763092, 'test-loss': 1.8404844999313354},
                   2: {'test-accuracy': 0.3953000009059906, 'test-loss': 2.322556495666504},
                   3: {'test-accuracy': 0.3382999897003174, 'test-loss': 2.6218838691711426}},
         'Res': {1: {'test-accuracy': 0.5180999636650085, 'test-loss': 1.7815804481506348},
                 2: {'test-accuracy': 0.48249998688697815, 'test-loss': 1.9527589082717896},
                 3: {'t est-accuracy': 0.4615999758243561, 'test-loss': 2.0412826538085938}}}

hello = {'Basic': {1: {'test-accuracy': 0.5030999779701233, 'test-loss': 1.8608421087265015},
                   2: {'test-accuracy': 0.38519999384880066, 'test-loss': 2.3547556400299072},
                   3: {'test-acc uracy': 0.3310000002384186, 'test-loss': 2.637925863265991}},
         'Res': {1: {'test-accuracy': 0.515999972820282, 'test-loss': 1.802215814590454},
                 2: {'test-accuracy': 0.47689998149871826, 'test-loss': 1.9512704610824585},
                 3: {'test-accuracy': 0.45489999651908875, 'test-loss': 2.055720090866089}},
         'Incep': {1: {'test-accuracy': 0.48019999265670776, 'te st-loss': 1.9830546379089355},
                   2: {'test-accuracy': 0.3944000005722046, 'test-loss': 2.3334555625915527},
                   3: {'test-accuracy': 0.32850000262260437, 'test-loss': 2.672494649887085}},
         'IncepRes': {1: {'test-accuracy': 0.49399998784065247, 'test-loss': 1.9076372385025024},
                      2: {'test-accuracy': 0.4311999976634979, 'test-loss': 2.1982483863830566},
                      3: {'test-accuracy': 0.405599981546402, 'test-loss': 2.3084638118743896}}}

hello = {'Basic': {1: {'test-accuracy': 0.5062999725341797, 'test-loss': 1.8494235277175903},
                   2: {'test-accuracy': 0.4007999897003174, 'test-loss': 2.3715624809265137},
                   3: {'test-accuracy': 0.3553999960422516, 'test - loss': 2.5369253158569336}},
         'Res': {1: {'test-accuracy': 0.513700008392334, 'test-loss': 1.7969074249267578},
                 2: {'test-accuracy': 0.4885999858379364, 'test-loss': 1.9147710800170898},
                 3: {'test-accuracy': 0.4381999969482422, 'test-loss': 2.1229088306427}},
         'Incep': {1: {'test-accuracy': 0.49129998683929443, 'test- loss': 1.9299565553665161},
                   2: {'test - accuracy': 0.39499998092651367, 'test - loss ': 2.386662483215332},
                   3: {'test - accuracy': 0.32839998602867126, 'test - loss': 2.7051334381103516}},
         'IncepRes': {1: {'test-accuracy': 0.49480000138282776, 'test-loss': 1.9374545812606812},
                      2: {'test-accuracy': 0.4260999858379364, 'test-loss': 2.2217910289764404},
                      3: {'t est - accuracy': 0.4027999937534332, 'test - loss': 2.3364388942718506}}}

hello = {'Incep': {1: {'test-accuracy': 0.7410999536514282, 'test-loss': 0.7501242756843567},
                   2: {'test-accuracy': 0.666700005531311, 'test-loss': 0.9484413862228394},
                   3: {'test-accur acy': 0.6189000010490417, 'test-loss': 1.081337332725525}},
         'IncepRes': {1: {'test-accuracy': 0.746399998664856, 'test-loss': 0.7463325262069702},
                      2: {'test-accuracy': 0.7060999870300293, 'test-loss': 0.839226245880127},
                      3: {'test-accuracy': 0.6876999735832214, 'test-loss': 0.947816789150238}},
         'Basic': {1: {'test-accuracy': 0.7633999586105347, 'te st-loss': 0.6751044392585754},
                   2: {'test-accuracy': 0.7277999520301819, 'test-loss': 0.7679538130760193},
                   3: {'test-accuracy': 0.6761999726295471, 'test-loss': 0.9415991902351379}},
         'Res': {1: {'test-accuracy': 0.7849000096321106, 'test-loss': 0.6318141222000122},
                 2: {'test-accuracy': 0.7507999539375305, 'test-loss': 0.7477369904518127},
                 3: {'test- accuracy': 0.7279999852180481, 'test-loss': 0.807895839214325}}}
