import matplotlib.pyplot as plt
import numpy as np
import pathlib
from utils import SCRIPTS_PATH

plt.style.use(SCRIPTS_PATH / 'style.mplstyle')


def lossaccfigure(history: np.ndarray, path: str, num_classes=None, history_std=None):
    x = range(history.shape[0])
    fig = plt.figure(figsize=(7, 4))
    ax0 = fig.add_subplot(121, title="loss")
    ax0.plot(history[:, 0], '.-', label='train')
    ax0.plot(history[:, 1], '*-', label='val')
    if type(history_std) == np.ndarray:
        ax0.fill_between(x, history[:, 0] - history_std[:, 0],
                         history[:, 0] + history_std[:, 0], alpha=0.2)
        ax0.fill_between(x, history[:, 1] - history_std[:, 1],
                         history[:, 1] + history_std[:, 1], alpha=0.2)
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('loss')
    ax0.legend()

    ax1 = fig.add_subplot(122, title="accuracy")
    ax1.plot(history[:, 2], '.-', label='train')
    ax1.plot(history[:, 3], '*-', label='val')
    if type(history_std) == np.ndarray:
        ax1.fill_between(x, history[:, 2] - history_std[:, 2],
                         history[:, 2] + history_std[:, 2], alpha=0.2)
        ax1.fill_between(x, history[:, 3] - history_std[:, 3],
                         history[:, 3] + history_std[:, 3], alpha=0.2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy (%)')
    if (num_classes):
        ax1.hlines(y=(1/num_classes) * 100, xmin=0,
                   xmax=history.shape[0], linewidth=2, colors='black', linestyles='dashed')

    ax1.legend()
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def replotAllResults():
    resultsPath = pathlib.Path.cwd() / 'results'
    historyPaths = list(resultsPath.rglob('*history.csv'))
    for historyPath in historyPaths:
        history = np.genfromtxt(historyPath, delimiter=',')
        lossaccfigure(history, historyPath.parent /
                      'train_lossacc.png', num_classes=5)


def replotResults(model, target):
    resultsPath = pathlib.Path.cwd() / 'results' / model / 's1' / target
    history = np.genfromtxt(resultsPath / 'history.csv', delimiter=',')
    history_std = np.genfromtxt(resultsPath / 'history_std.csv', delimiter=',')
    lossaccfigure(history, resultsPath / 'train_lossacc.png', 5, history_std)


if __name__ == "__main__":
    replotResults('ACT_40_Mini', 'ACT_40_Mini')
