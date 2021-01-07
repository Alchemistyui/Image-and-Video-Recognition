from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

import os
import pandas as pd
import numpy as np
# from sklearn.model_selection import StratifiedKFold

# import matplotlib.pyplot as plt
# from IPython import display
# %matplotlib inline
# display.set_matplotlib_formats('svg')


class NumeralDataset(Dataset):
    def __init__(self, trans):
        super(NumeralDataset, self).__init__()
        self.trans = trans
        with open(imglist) as f:
            self.img_list = [x.strip() for x in f]

        # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # y = np.array([0, 0, 1, 1])
        # skf = StratifiedKFold(n_splits=7, shuffle=True)
        # skf.get_n_splits(X, y)

        # for train_index, test_index in skf.split(X, y):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]

        print('load {} image'.format(len(self.img_list)))

    def __getitem__(self, idx):
        img = Image.open(self.root + self.img_list[index]).convert('RGB')
        img = self.trans(img)
        return img, label

    def __len__(self):
        return len(self.img_list)








