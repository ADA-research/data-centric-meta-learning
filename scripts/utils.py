from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib as plt

cwd = Path.cwd()
DATA_PATH = cwd / 'data'
SCRIPTS_PATH = cwd / 'scripts'


class Paths:
    def __init__(self, exp_path: Path):
        self.MODELS_PATH = exp_path / 'models'
        self.TRANSFERS_PATH = exp_path / 'transfers'
        self.FIGURES_PATH = exp_path / 'figures'
        self.FEATURES_PATH = exp_path / 'features'


DOMAIN_ORDER = [
    "large animals",
    "small animals",
    "plants",
    "plant diseases",
    "microscopy",
    "remote sensing",
    "vehicles",
    "manufacturing",
    "human actions",
    "ocr"
]

META_ALBUM_DATASETS = [
    # large animals
    'BRD_Mini',
    'DOG_Mini',
    'AWA_Mini',

    # small animals
    'PLK_Mini',
    'INS_2_Mini',
    'INS_Mini',

    # plants
    'FLW_Mini',
    'PLT_NET_Mini',
    'FNG_Mini',

    'PLT_VIL_Mini',
    'MED_LF_Mini',
    'PLT_DOC_Mini',

    'BCT_Mini',
    'PNU_Mini',
    'PRT_Mini',

    'RESISC_Mini',
    'RSICB_Mini',
    'RSD_Mini',

    'CRS_Mini',
    'APL_Mini',
    'BTS_Mini',

    'TEX_Mini',
    'TEX_DTD_Mini',
    'TEX_ALOT_Mini',

    'SPT_Mini',
    'ACT_40_Mini',
    'ACT_410_Mini',

    'MD_MIX_Mini',
    'MD_5_BIS_Mini',
    'MD_6_Mini'
]

DATASETS_SHORT = [dataset.replace("_Mini", "")
                  for dataset in META_ALBUM_DATASETS]


def unique_cols(df: pd.DataFrame):
    a: np.ndarray = df.to_numpy()
    return (a[0] == a).all(0)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_bars = len(data)

    bar_width = total_width / n_bars

    bars = []

    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width *
                         single_width, color=colors[i % len(colors)])

        bars.append(bar[0])

    if legend:
        ax.legend(bars, data.keys())
