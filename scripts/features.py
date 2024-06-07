# https://christophm.github.io/interpretable-ml-book/cnn-features.html
# https://www.sciencedirect.com/science/article/pii/S1361841522001177
# https://distill.pub/2017/feature-visualization/

import argparse
import time
from pathlib import Path
import pandas as pd
from utils import DOMAIN_ORDER, DATA_PATH, META_ALBUM_DATASETS
import torch
from dataset import MetaAlbumDataset
from torch.utils.data import DataLoader
import csv


channels = ['r', 'g', 'b']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='which dataset to compute features of.')
    parser.add_argument('--exp_path', type=str, required=True,
                        help='path of the experiment collection.')
    parser.add_argument('--num_target_classes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed to use.')
    parser.add_argument('--folds', type=int, default=5)

    return parser.parse_known_args()


def calculate_features(args):
    exp_path = Path(args.exp_path)
    features_path = exp_path / 'features' / f's{args.seed}' / args.dataset
    train_features_path = features_path / 'train'
    Path.mkdir(train_features_path, parents=True, exist_ok=True)

    test_features_path = features_path / 'test'
    Path.mkdir(test_features_path, parents=True, exist_ok=True)

    dataset_path = DATA_PATH / args.dataset
    meta_train_dataset = MetaAlbumDataset(
        dataset_path, args.seed, mode='train')

    dict_to_csv(pixel_features(meta_train_dataset),
                train_features_path / 'pixel.csv')

    dict_to_csv(global_features(meta_train_dataset),
                train_features_path / 'global.csv')

    dict_to_csv(achieved_train_acc(args.dataset, exp_path,
                args.seed), train_features_path / 'model.csv')

    for fold in range(args.folds):
        fold_path: Path = test_features_path / f'fold_{fold}'
        fold_path.mkdir(parents=True, exist_ok=True)
        test_dataset = MetaAlbumDataset(
            dataset_path, args.seed, mode='test', num_test_classes=5, fold_seed=fold)

        dict_to_csv(pixel_features(test_dataset), fold_path / 'pixel.csv')

        dict_to_csv(global_features(test_dataset, fold),
                    fold_path / 'global.csv')

        dict_to_csv(achieved_test_acc(args.dataset, exp_path,
                    args.seed, fold), fold_path / 'targets.csv')

        dict_to_csv(low_cost_proxy(args.dataset, exp_path,
                    args.seed, fold, 0), fold_path / 'proxy.csv')


def global_features(dataset: MetaAlbumDataset, fold=None):
    features = {}
    features['num_classes'] = len(dataset) / 40
    features['dataset_domain'] = DOMAIN_ORDER[META_ALBUM_DATASETS.index(
        dataset.name) // 3]
    features['dataset_name'] = dataset.name
    features['dataset_id'] = f'{dataset.name}_s{dataset.split_seed}'
    if fold != None:
        features['fold'] = str(fold)
        features['dataset_id'] += f'_f{fold}'
    features['seed'] = dataset.split_seed
    return features


def entropy(tensor):
    entropy = torch.zeros(tensor.shape[0], 1, requires_grad=True).cuda()
    imageSize = float(tensor.shape[1]*tensor.shape[2]*tensor.shape[3])
    for i in range(0, tensor.shape[0]):
        _, counts = torch.unique(tensor[i].data.flatten(
            start_dim=0), dim=0, return_counts=True)
        p = (counts)/imageSize
        entropy[i] = -torch.sum(p * torch.log2(p))
    return entropy.mean().item(), entropy.std().item()


def pixel_features(dataset: MetaAlbumDataset):
    start = time.time()
    features = {}
    dl = DataLoader(dataset, batch_size=len(dataset))
    for (input, _) in dl:
        input = input.to(device)
        features['pixel_global_mean'] = torch.mean(input).item()
        features['pixel_global_std'] = torch.std(input).item()
        for i, c in enumerate(channels):
            features[f'pixel_mean_{c}'] = torch.mean(input[:, i, :, :]).item()
            features[f'pixel_std_{c}'] = torch.std(input[:, i, :, :]).item()

        features['pixel_entropy_mean'], features['pixel_entropy_std'] = entropy(
            input)

        # implementation of Hasler and Susstrunk
        rg = torch.absolute(torch.subtract(
            input[:, 0, :, :], input[:, 1, :, :]))
        yb = torch.absolute(torch.subtract(torch.mul(
            torch.add(input[:, 0, :, :], input[:, 1, :, :]), 0.5), input[:, 2, :, :]))

        rg_mean, rg_std = torch.std_mean(rg, dim=[1, 2])
        yb_mean, yb_std = torch.std_mean(yb, dim=[1, 2])

        std_root = torch.sqrt(torch.pow(rg_std, 2) + torch.pow(yb_std, 2))
        mean_root = torch.sqrt(torch.pow(rg_mean, 2) + torch.pow(yb_mean, 2))

        colorfulness = torch.add(std_root, torch.mul(mean_root, 0.3))

        colorfulness_avg, colorfulness_std = torch.std_mean(colorfulness)

        features['pixel_colorfulness_avg'], features['pixel_colorfulness_std'] = colorfulness_avg.item(
        ), colorfulness_std.item()

    print(f'pixel features took {time.time() - start:.4f}s')
    return features


def low_cost_proxy(target_name, exp_path, seed, fold, epoch=0):
    start = time.time()
    proxies = {}
    for fmodel in META_ALBUM_DATASETS:
        df = pd.read_csv(exp_path / 'transfers' / f's{seed}' / fmodel / target_name /
                         f'history_f{fold}.csv', header=0, names=['t_loss', 'v_loss', 't_acc', 'v_acc'])
        proxies[f'{fmodel}_proxy'] = df.loc[epoch, 'v_acc']

    print(f'low cost proxy features took {time.time() - start:.4f}s')
    return proxies


def achieved_test_acc(target_name, exp_path, seed, fold, tail=10):
    accs = {}
    for fmodel in META_ALBUM_DATASETS:
        df = pd.read_csv(exp_path / 'transfers' / f's{seed}' / fmodel / target_name /
                         f'history_f{fold}.csv', header=0, names=['t_loss', 'v_loss', 't_acc', 'v_acc'])
        accs[f'{fmodel}_TARGET'] = df.loc[-tail:, 'v_acc'].mean()
    return accs


def achieved_train_acc(model_name, exp_path, seed, tail=10):
    accs = {}
    df = pd.read_csv(exp_path / 'models' / f's{seed}' / model_name /
                     'history.csv', header=0, names=['t_loss', 'v_loss', 't_acc', 'v_acc'])
    accs['train_acc'] = df.loc[-tail:, 'v_acc'].mean()
    return accs


def dict_to_csv(dict: dict, path: Path):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()
        w.writerow(dict)


if __name__ == '__main__':
    args, unparsed = parse_arguments()

    if len(unparsed) != 0:
        raise ValueError(f'Argument {unparsed} not recognized')

    print('features with the following arguments:', args)

    calculate_features(args)
