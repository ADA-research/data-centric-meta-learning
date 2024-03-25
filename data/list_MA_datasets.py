# This script lists all Meta-Album datasets available on OpenML

import openml

datasets_df = openml.datasets.list_datasets(output_format="dataframe")
MA_datasets = datasets_df[datasets_df['name'].str.contains(
    'Meta_Album\_.*\_Mini')]
print(MA_datasets)
