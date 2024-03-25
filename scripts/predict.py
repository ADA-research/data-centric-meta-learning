from pathlib import Path
from tmatrix import META_ALBUM_DATASETS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import SCRIPTS_PATH, bar_plot
import argparse
from dataframe import construct_dataframe
from tmatrix import plot_tmatrix
from collections import Counter


pixel_features = ['pixel_global_mean', 'pixel_mean_r', 'pixel_std_r', 'pixel_mean_g',
                  'pixel_std_g', 'pixel_mean_b', 'pixel_std_b', 'pixel_colorfulness_avg',
                  'pixel_colorfulness_std', 'pixel_entropy_mean', 'pixel_entropy_std']

feature_set_standard = set(['num_classes_train',
                            'pixel_global_mean_train', 'pixel_global_std_train',
                            'pixel_mean_r_train', 'pixel_std_r_train', 'pixel_mean_g_train',
                            'pixel_std_g_train', 'pixel_mean_b_train', 'pixel_std_b_train',
                            'pixel_entropy_mean_train', 'pixel_entropy_std_train',
                            'pixel_colorfulness_avg_train', 'pixel_colorfulness_std_train', 'pixel_global_mean_test',
                            'pixel_global_std_test', 'pixel_mean_r_test', 'pixel_std_r_test',
                            'pixel_mean_g_test', 'pixel_std_g_test', 'pixel_mean_b_test',
                            'pixel_std_b_test', 'pixel_entropy_mean_test', 'pixel_entropy_std_test',
                            'pixel_colorfulness_avg_test', 'pixel_colorfulness_std_test'])

feature_set_diff = set(['pixel_global_mean_diff', 'pixel_mean_r_diff', 'pixel_std_r_diff', 'pixel_mean_g_diff',
                        'pixel_std_g_diff', 'pixel_mean_b_diff', 'pixel_std_b_diff', 'pixel_colorfulness_avg_diff',
                        'pixel_colorfulness_std_diff', 'pixel_entropy_mean_diff', 'pixel_entropy_std_diff'])

feature_set_standard_proxy = feature_set_standard | set(['epoch_1_val'])

feature_set_diff_proxy = feature_set_diff | set(['epoch_1_val'])

feature_set_all = feature_set_standard_proxy | feature_set_diff_proxy


GLOB_IMPORTANCES = []

plt.style.use(SCRIPTS_PATH / 'style.mplstyle')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True,
                        help="path of the experiment collection.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed to use.")

    parser.add_argument('--exclude', action='store_true')
    parser.set_defaults(exclude=False)

    parser.add_argument('--feature_set', type=str, default='all')

    return parser.parse_known_args()


def transfer_matrix(full_df, path, target_name='TARGET', img_name='matrix', mode='absolute'):
    reversed = mode == 'absolute'

    df: pd.DataFrame = full_df[[
        'dataset_name_test', 'dataset_name_train', 'dataset_id_test', target_name]]
    df = df.rename(columns={target_name: 'TARGET'}, inplace=False)

    df['rank'] = df.groupby('dataset_id_test')['TARGET'].rank(
        method='dense', ascending=False).astype('int')
    df['pct_rank'] = df.groupby('dataset_id_test')['TARGET'].rank(
        method='dense', ascending=True, pct=True)

    df = df.groupby(['dataset_name_test', 'dataset_name_train'], as_index=False).agg(
        {'TARGET': ['mean', 'std'], 'rank': ['mean', 'std'], 'pct_rank': ['mean', 'std']})
    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    print(df)

    result_matrix = np.empty(
        (len(META_ALBUM_DATASETS), len(META_ALBUM_DATASETS)))
    string_values = [['NaN'] * len(META_ALBUM_DATASETS)
                     for i in range(len(META_ALBUM_DATASETS))]

    for fm_idx, fmodel in enumerate(META_ALBUM_DATASETS):
        for t_idx, target in enumerate(META_ALBUM_DATASETS):
            selection = df.loc[(df['dataset_name_test'] == target) & (
                df['dataset_name_train'] == fmodel)]
            if selection.empty:
                result_matrix[fm_idx, t_idx] = 0 if mode == 'absolute' else 30
            else:
                if mode == 'absolute':
                    val, std = selection['TARGET_mean'], selection['TARGET_std']
                    result_matrix[fm_idx, t_idx] = val
                    string_values[fm_idx][t_idx] = f'{int(val)}±{int(std)}'
                elif mode == 'ranked':
                    val, std = float(selection['rank_mean']), float(
                        selection['rank_std'])
                    result_matrix[fm_idx, t_idx] = val
                    string_values[fm_idx][t_idx] = f'{val:.1f}±{int(std)}'
                elif mode == 'relative':
                    val, std = selection['pct_rank_mean'] * \
                        100, selection['pct_rank_std'] * 100
                    result_matrix[fm_idx, t_idx] = val
                    string_values[fm_idx][t_idx] = f'{int(val)}±{int(std)}'

    result_matrix[np.isnan(result_matrix)] = 0

    plot_tmatrix(result_matrix, string_values, name=img_name,
                 store_path=path, reversed=reversed)


def print_rmse(name, rmse_train, rmse_test):
    print(f'{name} train rmse: {rmse_train} -- test rmse: {rmse_test}')


def rf(x_train: pd.DataFrame, y_train, x_test: pd.DataFrame, y_test, feature_set=feature_set_standard):
    rf = RandomForestRegressor(n_estimators=500, max_features=0.5, max_depth=7)

    x_train_numeric = x_train[list(feature_set)].select_dtypes('number')
    x_test_numeric = x_test[list(feature_set)].select_dtypes('number')
    rf.fit(x_train_numeric, y_train)

    pred_train = rf.predict(x_train_numeric)
    rmse_train = mean_squared_error(y_train, pred_train, squared=False)

    pred_test = pd.Series(rf.predict(x_test_numeric), index=y_test.index)
    rmse_test = mean_squared_error(y_test, pred_test, squared=False)

    print_rmse('random forest', rmse_train, rmse_test)

    importances = rf.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=x_train_numeric.columns)

    GLOB_IMPORTANCES.append(forest_importances)

    return pred_test


def bs_proxy(x_train, y_train, x_test, y_test, **kwargs):
    return x_test['epoch_1_val']


def bs_ma(x_train, y_train, x_test, y_test, **kwargs):
    full_df_train: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    full_df_test: pd.DataFrame = pd.concat([x_test, y_test], axis=1)
    pred = full_df_train.groupby('dataset_name_train')[
        'TARGET'].mean().rename('pred')

    pred_train = full_df_train.join(pred, on='dataset_name_train')
    pred_test = full_df_test.join(pred, on='dataset_name_train')

    rmse_train = mean_squared_error(
        pred_train['TARGET'], pred_train['pred'], squared=False)
    rmse_test = mean_squared_error(
        pred_test['TARGET'], pred_test['pred'], squared=False)

    print_rmse('baseline model average', rmse_train, rmse_test)

    return pred_test['pred']


def select_features(full_df: pd.DataFrame, fset: str) -> pd.DataFrame:
    loc_df = full_df.copy()
    loc_df = loc_df[['num_classes_train', 'dataset_name_test', 'dataset_name_train', 'fold',
                     'pixel_global_mean_train', 'pixel_global_std_train',
                     'pixel_mean_r_train', 'pixel_std_r_train', 'pixel_mean_g_train',
                     'pixel_std_g_train', 'pixel_mean_b_train', 'pixel_std_b_train',
                     'pixel_entropy_mean_train', 'pixel_entropy_std_train',
                     'pixel_colorfulness_avg_train', 'pixel_colorfulness_std_train', 'pixel_global_mean_test',
                     'pixel_global_std_test', 'pixel_mean_r_test', 'pixel_std_r_test',
                     'pixel_mean_g_test', 'pixel_std_g_test', 'pixel_mean_b_test',
                     'pixel_std_b_test', 'pixel_entropy_mean_test', 'pixel_entropy_std_test',
                     'pixel_colorfulness_avg_test', 'pixel_colorfulness_std_test', 'epoch_1_val', 'TARGET']]

    return loc_df


def predict(args):
    t_start = time.time()

    seeds = [1, 2, 3]
    exp_path = Path(args.exp_path)
    figures_path = exp_path / 'figures'
    figures_path.mkdir(exist_ok=True)

    features_dfs = []

    for seed in seeds:
        features_path = exp_path / 'features' / \
            f's{seed}' / 'full_features.csv'
        if features_path.is_file():
            features_dfs.append(pd.read_csv(features_path, index_col=0))
        else:
            print("full df not found. Reconstructing...")
            features_dfs.append(construct_dataframe(
                exp_path / 'features', seed))
            print(f'construction: {time.time() - t_start}s')

    df = pd.concat(features_dfs, axis=0)

    df['fold'] = df['fold'].astype(str)
    df = df.dropna(subset=['TARGET']).reset_index()
    df['model_avg_acc'] = df.groupby(['dataset_name_train'])[
        'TARGET'].transform(np.mean)

    print(df.info(verbose=True))

    features = select_features(df, args.feature_set)
    print('final (numeric) features: ', features.select_dtypes('number').columns)

    X = df.drop('TARGET', axis=1)
    y = df['TARGET']

    models = {
        'rf standard': {'func': rf, 'feature_set': feature_set_standard},
        'rf diff': {'func': rf, 'feature_set': feature_set_diff},
        'rf standard proxy': {'func': rf, 'feature_set': feature_set_standard_proxy},
        'rf diff proxy': {'func': rf, 'feature_set': feature_set_diff_proxy},
        'rf all': {'func': rf, 'feature_set': feature_set_all},
        'model average': {'func': bs_ma, 'feature_set': None},
        'low cost proxy': {'func': bs_proxy, 'feature_set': None},
    }

    for model_name in models.keys():
        df[f'{model_name}_pred'] = np.nan

    for i, test_ds in enumerate(META_ALBUM_DATASETS[:]):

        # making a leave-one-out split
        train_idx = df.index[(df['dataset_name_test'] != test_ds)].to_list()
        test_idx = df.index[(df['dataset_name_test'] == test_ds)].to_list()

        print()
        print(
            f'fold {i} testing {test_ds}: #train {len(train_idx)}, #test {len(test_idx)}')

        for model_name, model_dict in models.items():
            # make a prediction using the given model
            test_pred = model_dict['func'](
                X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx], feature_set=model_dict['feature_set'])
            df.loc[test_idx, f'{model_name}_pred'] = test_pred.values

    if args.exclude:
        df = df.drop(df[(df['dataset_name_test'] == test_ds) &
                     (df['dataset_name_train'] != test_ds)].index)

    df.to_csv(exp_path / 'meta-data-predictions.csv')

    df = df.sort_values(['dataset_id_test', 'TARGET'],
                        ascending=[False, False])
    df['rank'] = df.groupby('dataset_id_test')['TARGET'].rank(
        ascending=False).astype('int')
    df['best_acc'] = df.groupby('dataset_id_test')['TARGET'].transform(max)
    df['loss'] = np.abs(df['TARGET'] - df['best_acc'])

    avg_rank_df = df.groupby(['dataset_name_train'])[
        'rank'].agg(['mean', 'std'])
    fig, ax = plt.subplots()
    avg_rank_df.plot.bar(y='mean', yerr='std', ax=ax)
    ax.set_ylabel('average rank')
    fig.tight_layout()
    fig.savefig(figures_path / f'model_rankings.png', dpi=300)

    print(df.info(verbose=True))

    fig_rmse_per_dataset, ax_rmse_per_dataset = plt.subplots()
    ranks = ['best', 'top 2', 'top 3', 'top 4', 'top 5']
    rank_positions = np.arange(len(ranks))
    dataset_positions = np.arange(len(META_ALBUM_DATASETS))
    bar_width = 0.15

    losses = {}
    rmse = {}
    rmse_ds: pd.DataFrame

    rank_bar_scores = {}

    for i, model_name in enumerate(models.keys()):
        transfer_matrix(
            df, figures_path, target_name=f'{model_name}_pred', img_name=model_name+'_tmatrix')
        transfer_matrix(
            df, figures_path, target_name=f'{model_name}_pred', img_name=model_name+'_tmatrix_ranked', mode='ranked')

        selected_idx = df.groupby(['dataset_id_test'])[
            f'{model_name}_pred'].transform(max) == df[f'{model_name}_pred']
        selections = df[selected_idx].sort_values(
            'TARGET', ascending=True).drop_duplicates(['dataset_id_test'])
        losses[model_name] = (selections['TARGET'] -
                              selections[f'best_acc']).abs()
        rmse[model_name] = mean_squared_error(
            df['TARGET'], df[f'{model_name}_pred'], squared=False)

        rmse_ds = selections.groupby(['dataset_name_test']).apply(lambda x: mean_squared_error(
            x['TARGET'], x[f'best_acc'], squared=False)).reset_index(name='rmse')
        ax_rmse_per_dataset.bar(dataset_positions + (i - len(models) // 2)
                                * bar_width, rmse_ds['rmse'], bar_width, label=model_name)

        cntr = Counter(selections['rank'].values)
        counts = [cntr[rank+1] for rank in range(len(ranks))]
        cumsum = np.cumsum(counts)
        rank_bar_scores[model_name] = cumsum

    selection_diag = df[df['dataset_name_test'] == df['dataset_name_train']]
    losses['diagonal'] = (selection_diag['TARGET'] -
                          selection_diag[f'best_acc']).abs()
    cntr = Counter(selection_diag['rank'].values)
    counts = [cntr[rank+1] for rank in range(len(ranks))]
    cumsum = np.cumsum(counts)
    rank_bar_scores['diagonal'] = cumsum

    num_preds = df['dataset_id_test'].nunique()
    rand_expected = np.full((len(ranks)), num_preds *
                            1 / (len(META_ALBUM_DATASETS) - (1 * args.exclude)))
    rand_expected = np.cumsum(rand_expected)
    rank_bar_scores['random expected'] = rand_expected

    fig, ax = plt.subplots()
    bar_plot(ax, rank_bar_scores)
    ax.set_xticks(rank_positions, ranks)
    ax.set_ylabel(f'number of predictions out of {num_preds}')
    ax.set_xlabel('true rank of prediction')
    fig.tight_layout()
    fig.savefig(figures_path / 'rank_histogram.png', dpi=300)

    ax_rmse_per_dataset.set_xticks(dataset_positions, [name.replace(
        '_Mini', '') for name in rmse_ds['dataset_name_test'].tolist()], rotation=90)
    ax_rmse_per_dataset.set_ylabel('rmse')
    ax_rmse_per_dataset.set_xlabel('test task dataset')
    ax_rmse_per_dataset.legend()
    fig_rmse_per_dataset.tight_layout()
    fig_rmse_per_dataset.savefig(
        figures_path / 'rmse_per_dataset.png', dpi=300)

    losses['random'] = df.groupby('dataset_id_test')['loss'].agg('mean')
    losses_df = pd.DataFrame(losses)

    fig, ax = plt.subplots()
    losses_df.boxplot(ax=ax, whis=(5, 95), showfliers=False)
    ax.set_ylabel('loss in accuracy in %')
    ax.set_xticks(ax.get_xticks(), rotation=45)
    fig.tight_layout()
    fig.savefig(figures_path / f'prediction_model_losses.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(*zip(*rmse.items()))
    ax.set_ylabel('rmse in %')
    fig.tight_layout()
    fig.savefig(figures_path / f'prediction_model_rmse.png', dpi=300)

    print(f'prediction: {time.time() - t_start}s')


if __name__ == '__main__':
    args, unparsed = parse_arguments()

    if len(unparsed) != 0:
        raise ValueError(f'Argument {unparsed} not recognized')

    predict(args)
