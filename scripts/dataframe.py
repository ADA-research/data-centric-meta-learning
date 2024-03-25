from utils import META_ALBUM_DATASETS
from pathlib import Path
import pandas as pd


def construct_dataframe(path: Path, seed):
    features_path = path / f's{seed}'
    full_features = []

    for fmodel in META_ALBUM_DATASETS:
        for target in META_ALBUM_DATASETS:
            train_features_files = list(
                Path.glob(features_path / fmodel / 'train', '*.csv'))
            test_features_folders = [path for path in Path.glob(
                features_path / target / 'test', '*/') if path.is_dir()]

            if not train_features_files or not test_features_folders:
                continue

            train_features = []
            for file in train_features_files:
                df = pd.read_csv(file, header=0)
                train_features.append(df)

            train_df = pd.concat(train_features, axis=1)

            for folder in test_features_folders:
                test_features = []
                test_features_files = list(Path.glob(folder, '*.csv'))

                if not test_features_files:
                    print(f'no files found in: {folder}')
                    continue

                for file in test_features_files:
                    df = pd.read_csv(file, header=0)
                    test_features.append(df)

                test_df = pd.concat(test_features, axis=1)

                # only keep accuracy that was achieved for this combination
                test_df = test_df.rename(
                    columns={f'{fmodel}_TARGET': 'TARGET'})
                test_df = test_df.loc[:, ~
                                      test_df.columns.str.endswith('_TARGET')]

                test_df = test_df.rename(
                    columns={f'{fmodel}_proxy': 'epoch_1_val'})
                test_df = test_df.loc[:, ~
                                      test_df.columns.str.endswith('_proxy')]

                full_features.append(train_df.join(
                    test_df, lsuffix='_train', rsuffix='_test'))

    full_df = pd.concat(full_features, axis=0, ignore_index=True)
    full_df['dataset_id_train'] = full_df['dataset_name_train'] + f'_s{seed}'
    full_df.to_csv(features_path / f'full_features.csv')
    return full_df
