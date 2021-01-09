import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict as edict

from utils import AverageMeter
from elastic_weight_consolidation import ElasticWeightConsolidation

class KFoldSolver(object):
    """docstring for KFoldSolver"""
    def __init__(self, params):
        super(KFoldSolver, self).__init__()
        self.params = params
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=net.parameters(), lr= self.params.lr, weight_decay=weight_decay)


    def train(self, net, train_loader):
        epoch_losses = AverageMeter()
        epoch_accs = AverageMeter()
        for epo in range(self.params.epochs):
            net.train()
            losses = AverageMeter()
            corrects = []
            for batchID, (img, label) in enumerate(train_loader):
                img, label = input 
                output = net(img)

                optim.zero_grad()
                loss = self.loss_func(output, label)
                loss.backward()
                optim.step()
                corrects += output.eq(label).tolist()

                losses.update(loss.item(), img.size(0))
                accs.update(corrects)
                del loss, output

            acc = sum(corrects) / float(len(corrects))
            epoch_losses.update(losses.avg)
            print(f'Epoch {epo:2d}: train loss {losses.avg:.4f}, train acc {acc:.4f}')
        return epoch_losses.avg, epoch_accs.avg

    def val(self, net, val_loader):
        net.eval()
        losses = AverageMeter()
        corrects = []
        for batchID, (img, label) in enumerate(val_loader):
            img, label = input 
            output = net(img)

            loss = self.loss_func(output, target)
            corrects += output.eq(label).tolist()
            losses.update(loss.item(), img.size(0))
            del loss, output
        acc = sum(corrects) / float(len(corrects))
        return losses.avg, acc




    def k_fold_training(self, imgs, labels):
        # returns stratified folds that are made by preserving the percentage of samples for each class.
        skf = StratifiedKFold(n_splits=7, shuffle=True)

        # 不需要变更dataset
        trans = transforms.Compose([
            transforms.Resize((410, 410)),# 缩放
            transforms.RandomCrop(400), #裁剪
            transforms.ToTensor(), # change to tensor and normalize to 0-1
            # transforms.Normalize(norm_mean, norm_std),# 标准化
        ])
        for fold, (train_idx, test_idx) in enumerate(skf.split(imgs, labels)):
            print('-'*80)
            print('Fold {}'.format(fold))
            img_train, img_test = imgs.iloc[train_idx], imgs.iloc[test_idx]
            label_train, label_test = labels.iloc[train_idx], labels.iloc[test_idx]

            train_loader = DataLoader(
                dataset=NumeralDataset(img_train, label_train, trans),
                shuffle=True,
                batch_size=self.params.batchsize,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            test_loader = DataLoader(
                dataset=NumeralDataset(img_test, label_test, trans),
                shuffle=False,
                batch_size=self.params.batchsize,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            fold_train_loss, fold_test_acc = self.train(net, train_loader, test_loader)

            # ?最后的模型应该取啥啊

            print(fold_train_loss, fold_test_acc)




def main(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna()

    imgs=df
    labels = df['Class']

    params = edict({'epochs': 10, 'batchsize': 8, 'lr': 1e-3, 'num_classes': 10})
    k_fold_training(imgs, labels, params)
    

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)


if __name__ == "__main__":
    csv_file = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen.csv'
    main()








