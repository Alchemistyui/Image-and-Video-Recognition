from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

import os
# import pandas as pd
# import numpy as np
from PIL import Image
# from sklearn.model_selection import StratifiedKFold

# import matplotlib.pyplot as plt
# from IPython import display
# %matplotlib inline
# display.set_matplotlib_formats('svg')


class NumeralDataset(Dataset):
    def __init__(self, df, trans, num_cls=10):
        super(NumeralDataset, self).__init__()
        self.root = '/Users/Alchemist/Desktop/final_project/data/dataset/'
        self.trans = trans
        self.img_list = df['FileName'].tolist()
        self.cls_label = df['Class'].tolist()
        self.num_cls = num_cls

        print('load {} image'.format(len(self.img_list)))

    def __getitem__(self, idx):
        img = Image.open(self.root + self.img_list[idx]).convert('RGB')
        img = self.trans(img)
        label = self.cls_label[idx]
        # label = torch.eye(self.num_cls)[int(label[-1]),:]
        label = int(label[-1])
        return img, label

    def __len__(self):
        return len(self.img_list)

    def label2onehot(self, labels):
        labels = [int(labels[n][-1]) for n in labels]
        return torch.eye(self.num_cls)[labels,:]








