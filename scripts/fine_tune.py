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
    parser.add_argument("--foundation", type=str, required=True,
                        help="which foundation dataset to use.")
    parser.add_argument("--target", type=str, required=True,
                        help="which target dataset to tranfer to.")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="path of the experiment collection.")
    parser.add_argument("--num_target_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50,
                        help="how many training epochs to use.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed to use.")
    parser.add_argument("--folds", type=int, default=5,
                        help="amount of target tasks to transfer to")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for optimizer.")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="momentum for optimizer.")
    parser.add_argument("--test_size", type=float,
                        default=0.2, help="train validation ratio.")
    parser.add_argument("--architecture", type=str, default='resnet-18',
                        choices=['resnet-10', 'resnet-18', 'resnet-34'])

    return parser.parse_known_args()


def fine_tune(args):
    print(f'transfering {args.foundation} to {args.target}')

    starttime = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_path = Path(args.exp_path)
    target_path = DATA_PATH / args.target
    model_path = exp_path / 'models' / f's{args.seed}' / args.foundation
    transfer_path = exp_path / 'transfers' / \
        f's{args.seed}' / args.foundation / args.target
    Path.mkdir(transfer_path, exist_ok=True, parents=True)

    for f in range(args.folds):
        dataset = MetaAlbumDataset(target_path, args.seed, mode='test',
                                   num_test_classes=args.num_target_classes, fold_seed=f)
        train_idx, val_idx = train_test_split(np.arange(len(
            dataset)), test_size=0.2, shuffle=True, stratify=dataset.targets, random_state=0)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(
            dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_dl = DataLoader(
            dataset, batch_size=args.batch_size, sampler=val_sampler)

        if args.architecture == 'resnet-10':
            print('using resnet 10')
            model_obj = ResNet(10, device, dataset.num_classes)
        elif args.architecture == 'resnet-18':
            print('using resnet 18')
            model_obj = ResNet(18, device, dataset.num_classes)
        elif args.architecture == 'resnet-34':
            print('using resnet 34')
            model_obj = ResNet(34, device, dataset.num_classes)

        model_dict = torch.load(model_path / "model.pth")
        model_obj.load_params(model_dict)
        model_obj.freeze_layers(True, dataset.num_classes)
        optimizer = torch.optim.SGD(
            model_obj.parameters(), args.lr, momentum=args.momentum)
        model_obj = model_obj.to(device)
        criterion = nn.CrossEntropyLoss()

        history = np.empty((args.epochs, 4))

        for e in range(args.epochs):
            history[e, 0], history[e, 2] = epoch(
                model_obj, device, train_dl, optimizer, criterion, False, e)
            history[e, 1], history[e, 3] = epoch(
                model_obj, device, val_dl, optimizer, criterion, True, e)

        np.savetxt(transfer_path / f'history_f{f}.csv', history,
                   delimiter=",", header='t_loss,v_loss,t_acc,v_acc', comments='')

        lossaccfigure(history, transfer_path /
                      f'train_lossacc_f{f}.png', dataset.num_classes)

    runtime = time.time() - starttime
    print(f"total duration of transfer: {runtime:.4f}s")


if __name__ == "__main__":
    args, unparsed = parse_arguments()

    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")

    print("transfer with the following arguments:", args)

    fine_tune(args)
