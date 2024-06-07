import argparse
import time
from resnet import ResNet
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
from pathlib import Path
import numpy as np
from dataset import MetaAlbumDataset
from figures import lossaccfigure
from epoch import epoch
from utils import DATA_PATH


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="which foundation dataset to use.")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="path of the experiment collection.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="how many training epochs to use.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for random processes.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for optimizer.")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="momentum for optimizer.")
    parser.add_argument("--test_size", type=float,
                        default=0.2, help="momentum for optimizer.")
    parser.add_argument("--architecture", type=str, default='resnet-18',
                        choices=['resnet-10', 'resnet-18', 'resnet-34'])

    return parser.parse_known_args()


def pre_train(args):
    starttime = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_path = Path(args.exp_path)
    data_path = DATA_PATH / args.dataset
    fmodel_path = exp_path / 'models' / f's{args.seed}' / args.dataset
    Path.mkdir(fmodel_path, exist_ok=True, parents=True)

    dataset = MetaAlbumDataset(data_path, args.seed, mode='train')
    train_idx, val_idx = train_test_split(np.arange(len(
        dataset)), test_size=args.test_size, shuffle=True, stratify=dataset.targets, random_state=0)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_dl = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_dl = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=val_sampler)

    if args.architecture == 'resnet-10':
        model_obj = ResNet(10, device, dataset.num_classes)
    elif args.architecture == 'resnet-18':
        model_obj = ResNet(18, device, dataset.num_classes)
    elif args.architecture == 'resnet-34':
        model_obj = ResNet(34, device, dataset.num_classes)

    model_obj = model_obj.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model_obj.parameters(), lr=args.lr, momentum=args.momentum)

    history = np.empty((args.epochs, 4))

    for e in range(args.epochs):
        history[e, 0], history[e, 2] = epoch(
            model_obj, device, train_dl, optimizer, criterion, False, e)
        history[e, 1], history[e, 3] = epoch(
            model_obj, device, val_dl, optimizer, criterion, True, e)
        np.savetxt(fmodel_path / 'history.csv', history, delimiter=",",
                   header='t_loss,v_loss,t_acc,v_acc', comments='')

    runtime = time.time() - starttime
    print(f"total duration of training: {runtime:.4f}s")

    model_path = fmodel_path / "model.pth"
    torch.save(model_obj.state_dict(), model_path)

    lossaccfigure(history, fmodel_path /
                  'train_lossacc.png', dataset.num_classes)


if __name__ == "__main__":
    args, unparsed = parse_arguments()

    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")

    print("training with the following arguments:", args)

    pre_train(args)
