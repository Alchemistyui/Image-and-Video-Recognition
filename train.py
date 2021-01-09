import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict as edict

from utils import AverageMeter
from elastic_weight_consolidation import ElasticWeightConsolidation
from network import BaseModel
from dataset import NumeralDataset

class TrainSolver(object):
    """docstring for TrainSolver"""
    def __init__(self, params, train_path, test_path):
        super(TrainSolver, self).__init__()
        self.params = params
        self.train_path = train_path
        self.test_path = test_path
        loss_func = nn.CrossEntropyLoss()
        # self.optim = torch.optim.Adam(params=net.parameters(), lr= self.params.lr, weight_decay=weight_decay)
        self.net = ElasticWeightConsolidation(BaseModel(400*400, 100, 10), crit=loss_func, lr=1e-4)

    def load_data(self):
        trans = transforms.Compose([
            transforms.Resize((410, 410)),# 缩放
            transforms.RandomCrop(400), #裁剪
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), # change to tensor and normalize to 0-1
            # transforms.Normalize(norm_mean, norm_std),# 标准化
        ])

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # img_train, img_test = imgs.iloc[train_idx], imgs.iloc[test_idx]
        # label_train, label_test = labels.iloc[train_idx], labels.iloc[test_idx]
        self.trainset = NumeralDataset(train_df, trans)
        self.testset = NumeralDataset(test_df, trans)
        self.train_loader = DataLoader(
            dataset=self.trainset,
            shuffle=True,
            batch_size=self.params.batchsize,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        self.test_loader = DataLoader(
            dataset=self.testset,
            shuffle=False,
            batch_size=self.params.batchsize,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        # return train_loader, test_loader


    def train(self):
        self.load_data()

        epoch_losses = AverageMeter()
        epoch_accs = AverageMeter()
        for epo in range(self.params.epochs):
            # self.net.train()
            losses = AverageMeter()
            corrects = []
            for batchID, (img, label) in tqdm(enumerate(self.train_loader)):
                # img, label = input 
                # output = net(img)
                self.net.forward_backward_update(img, label)
                self.net.register_ewc_params(self.trainset, self.params.batchsize, 300)

            val_acc = self.val()
            print(f'Epoch {epo}: val acc {val_acc}')

        #         optim.zero_grad()
        #         loss = self.loss_func(output, label)
        #         loss.backward()
        #         optim.step()
        #         corrects += output.eq(label).tolist()

        #         losses.update(loss.item(), img.size(0))
        #         accs.update(corrects)
        #         del loss, output

        #     acc = sum(corrects) / float(len(corrects))
        #     epoch_losses.update(losses.avg)
        #     print(f'Epoch {epo:2d}: train loss {losses.avg:.4f}, train acc {acc:.4f}')
        # return epoch_losses.avg, epoch_accs.avg

    def val(self):
        self.net.eval()
        # losses = AverageMeter()
        corrects = [] 
        for batchID, (img, label) in enumerate(self.val_loader):
            # img, label = input 
            output = self.net(img)

            # loss = self.loss_func(output, target)
            corrects += output.eq(label).tolist()
            # losses.update(loss.item(), img.size(0))
            # del loss, output
        acc = sum(corrects) / float(len(corrects))
        # return losses.avg, acc
        return acc






def main():
    # df = pd.read_csv(csv_file)
    # df = df.dropna()
    train_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_train.csv'
    test_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_test.csv'
    params = edict({'epochs': 10, 'batchsize': 8, 'lr': 1e-3, 'num_classes': 10})

    solver = TrainSolver(params, train_path, test_path)

    solver.train()

  

if __name__ == "__main__":
    
    main()








