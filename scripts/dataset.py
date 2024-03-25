from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from sklearn import preprocessing
import math
import pandas as pd
from PIL import Image
import torch
import numpy as np
import json


class MetaAlbumDataset(Dataset):
    def __init__(self,
                 ds_path: str,
                 split_seed: int,
                 mode: str,
                 ratio=0.8,
                 num_test_classes=None,
                 fold_seed=0,
                 verbose=True,
                 ):
        super(MetaAlbumDataset, self).__init__()

        self.split_seed = split_seed
        rng_classes = np.random.RandomState(self.split_seed)

        self.path = Path(ds_path)
        self.name = self.path.stem
        self.info = pd.read_csv(ds_path / "labels.csv")
        all_classes = self.info.CATEGORY.unique()
        rng_classes.shuffle(all_classes)

        split = math.floor(ratio * len(all_classes))
        if mode == 'train':
            self.classes = all_classes[:split]
        elif mode == 'test':
            test_classes = all_classes[split:]
            rng_test_classes = np.random.RandomState(fold_seed)
            self.classes = rng_test_classes.choice(
                test_classes, min(num_test_classes, len(test_classes)), False)
        elif mode == 'full':
            self.classes = all_classes
        else:
            raise Exception("no existing dataset mode selected")

        if verbose:
            print("TRAIN CLASSES: ", all_classes[:split])
            print("TEST CLASSES: ", all_classes[split:])
            print("SELECTED CLASSES: ", self.classes)

        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.classes)

        self.instances = self.info[self.info.CATEGORY.isin(self.classes)]

        self.num_classes = len(self.classes)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __getitem__(self, i):
        img_info = self.instances.iloc[i]
        img_name = img_info.FILE_NAME
        img = Image.open(self.path / 'images' / img_name).convert('RGB')
        img = self.transform(img)

        img_label = img_info['CATEGORY']

        return img, torch.LongTensor(self.le.transform([img_label])).squeeze()

    def __len__(self):
        return len(self.instances)

    def getDatasetInfo(self):
        with open(self.path / "info.json") as json_file:
            info = json.load(json_file)
        return info

    @property
    def targets(self):
        return self.instances.CATEGORY.tolist()
