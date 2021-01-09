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
    def __init__(self, params):
        super(TrainSolver, self).__init__()
        self.params = params
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

        train_df = pd.read_csv(csv_files[0])
        test_df = pd.read_csv(csv_files[1])

        # img_train, img_test = imgs.iloc[train_idx], imgs.iloc[test_idx]
        # label_train, label_test = labels.iloc[train_idx], labels.iloc[test_idx]
        self.trainset = NumeralDataset(train_df, trans, self.params.data_root)
        self.testset = NumeralDataset(test_df, trans, self.params.data_root)
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

    def lifelong_training(self, path_list):
        num_task = len(path_list)

        for task in range(num_task):
            print('-'*80)
            print(path_list[task][0].split('.')[0])
            self.train(path_list[task])
            print('-'*80)




    def train(self, csv_files):
        self.load_data(csv_files)
        self.net.model.train()

        epoch_losses = AverageMeter()
        epoch_accs = AverageMeter()
        for epo in range(self.params.epochs):
            # self.net.train()
            losses = AverageMeter()
            corrects = []
            for batchID, (img, label) in tqdm(enumerate(self.train_loader)):
                # img, label = input 
                # output = net(img)
                loss = self.net.forward_backward_update(img, label)
                losses.update(loss.item(), img.size(0))
                del loss

            # print(f'Epoch {epo:2d}: Train loss {losses.avg:4f}')
            train_acc = self.eval(self.train_loader)
            print(f'Epoch {epo:2d}: Train loss {losses.avg:.4f}, Train acc {train_acc:.4f}')
            val_acc = self.eval(self.test_loader)
            print(f'Epoch {epo:2d}: Test acc {val_acc:4f}')

        self.net.register_ewc_params(self.trainset, self.params.batchsize, 30)


    def eval(self, dataloader):
        self.net.model.eval()
        # losses = AverageMeter()
        corrects = [] 
        acc = 0
        # import pdb
        # pdb.set_trace()
        for batchID, (img, label) in enumerate(dataloader):
            output = self.net.model(img)

            acc += (output.argmax(dim=1).long() == label).float().mean()

        return acc / len(dataloader)






def main():
    # df = pd.read_csv(csv_file)
    # df = df.dropna()
    path_perfix = '/Users/Alchemist/Desktop/final_project/data/csv_files/'
    path_list = [['Writer_Cheng_train.csv', 'Writer_Cheng_test.csv'], 
        ['Writer_Peng_train.csv', 'Writer_Peng_test.csv'],
        ['Writer_Shen_train.csv', 'Writer_Shen_test.csv'],
        ['Writer_Wang_train.csv', 'Writer_Wang_test.csv']]
    # train_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_train.csv'
    # test_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_test.csv'
    params = edict({'epochs': 15, 'batchsize': 8, 'lr': 1e-3, 'num_classes': 10, 
        'data_root': '/Users/Alchemist/Desktop/final_project/data/dataset/'})

    solver = TrainSolver(params)

    solver.train()

  

if __name__ == "__main__":
    
    main()








