import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import META_ALBUM_DATASETS, DATASETS_SHORT, SCRIPTS_PATH


plt.style.use(SCRIPTS_PATH / 'style.mplstyle')


def plot_tmatrix(result_matrix: np.ndarray,
                 string_values=None,
                 name='matrix',
                 store_path: Path = Path.cwd(),
                 reversed=True
                 ):

    resmat_list = result_matrix.tolist()

    if string_values == None:
        string_values = [[f'{int(a)}' for a in i] for i in resmat_list]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(18.5, 10.5)
    palette = sns.color_palette(
        'coolwarm' if reversed else 'coolwarm_r', as_cmap=True)
    sns.heatmap(
        result_matrix,
        annot=string_values,
        annot_kws={"fontsize": 8},
        fmt='',
        cmap=palette,
        cbar=False,
        ax=ax,
        xticklabels=DATASETS_SHORT,
        yticklabels=DATASETS_SHORT,
        linewidth=0.5
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_xticks(np.arange(0, 30, 3), minor=True)
    ax.set_yticks(np.arange(0, 30, 3), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.tick_params(axis='both', which='major', labelsize=10,
                   labelbottom=False, bottom=False, top=True, labeltop=True)
    ax.set_ylabel("foundation model")
    ax.set_xlabel("target dataset")
    ax.xaxis.set_label_position('top')
    fig.tight_layout()
    fig.savefig(store_path / f'{name}.png', dpi=300)
