import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision


from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from utils import AverageMeter
from elastic_weight_consolidation import ElasticWeightConsolidation
from network import BaseModel
from dataset import NumeralDataset

import pdb

class TrainSolver(object):
    """docstring for TrainSolver"""
    def __init__(self, params):
        super(TrainSolver, self).__init__()
        self.params = params
        loss_func = nn.CrossEntropyLoss()
        # self.optim = torch.optim.Adam(params=net.parameters(), lr= self.params.lr, weight_decay=weight_decay)
        self.net = ElasticWeightConsolidation(BaseModel(400*400, 100, 10), crit=loss_func, lr=self.params.lr)
        self.writer = SummaryWriter('log5')

    def load_data(self, csv_files):
        trans = transforms.Compose([
            transforms.Resize((410, 410)),# 缩放
            transforms.RandomCrop(400), #裁剪
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), # change to tensor and normalize to 0-1
            # transforms.Normalize(norm_mean, norm_std),# 标准化
        ])

        train_df = pd.read_csv(self.params.csv_root+csv_files[0])
        test_df = pd.read_csv(self.params.csv_root+csv_files[1])

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

    def train_base(self, csv_files):
        self.net = BaseModel(400*400, 100, 10)
        self.load_data(csv_files[0])
        
        self.loss_func = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=self.net.parameters(), lr= self.params.lr)

        # self.net.train()

        for epo in range(self.params.epochs):
            self.net.train()
            losses = AverageMeter()
            # corrects = []
            # acc = 0
            for batchID, (img, label) in enumerate(self.train_loader):
                output = self.net(img)

                optim.zero_grad()
                loss = self.loss_func(output, label)
                loss.backward()
                optim.step()
                # pdb.set_trace()
                # corrects += output.eq(label).tolist()


                losses.update(loss.item(), img.size(0))
                # accs.update(corrects)
                del loss, output

            train_acc = self.eval(self.train_loader, 'train')
            test_acc, test_loss = self.eval(self.test_loader, 'test')
            print(f'Epoch {epo:2d}: Train loss {losses.avg:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:4f}, Test loss {test_loss:4f}')
            self.writer.add_scalar('Train/Loss', losses.avg, epo)
            self.writer.add_scalar('Train/Acc', train_acc, epo)
            self.writer.add_scalar('Test/Acc', test_acc, epo)
            self.writer.add_scalar('Test/Loss', test_loss, epo)




    def train(self, csv_files):
        self.load_data(csv_files)
        self.net.model.train()

        epoch_losses = AverageMeter()
        epoch_accs = AverageMeter()
        for epo in range(self.params.epochs):
            # self.net.train()
            losses = AverageMeter()
            corrects = []
            # for batchID, (img, label) in tqdm(enumerate(self.train_loader)):
            for batchID, (img, label) in enumerate(self.train_loader):
                # img, label = input 
                # output = net(img)
                loss = self.net.forward_backward_update(img, label)
                losses.update(loss.item(), img.size(0))
                del loss

            # print(f'Epoch {epo:2d}: Train loss {losses.avg:4f}')
            train_acc = self.eval(self.train_loader, 'train')
            test_acc, test_loss = self.eval(self.test_loader, 'test')
            print(f'Epoch {epo:2d}: Train loss {losses.avg:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:4f}, Test loss {test_loss:4f}')
            self.writer.add_scalar('Train/Loss', losses.avg, epo)
            self.writer.add_scalar('Train/Acc', train_acc, epo)
            self.writer.add_scalar('Test/Acc', test_acc, epo)
            self.writer.add_scalar('Test/Loss', test_loss, epo)
            # print(f'Epoch {epo:2d}: Test acc {val_acc:4f}')

        self.net.register_ewc_params(self.trainset, self.params.batchsize, 30)


    def eval(self, dataloader, mode='test'):
        # self.net.model.eval()
        self.net.eval()

        if mode == 'train':
            corrects = [] 
            acc = 0
            for batchID, (img, label) in enumerate(dataloader):
                output = self.net(img)

                acc += (output.argmax(dim=1).long() == label).float().mean()

            return acc / len(dataloader)

        else:
            corrects = [] 
            acc = 0
            losses = AverageMeter()
            for batchID, (img, label) in enumerate(dataloader):
                output = self.net(img)
                loss = self.loss_func(output, label)
                acc += (output.argmax(dim=1).long() == label).float().mean()
                losses.update(loss.item(), img.size(0))                

            return acc / len(dataloader), losses.avg






def main():
    # df = pd.read_csv(csv_file)
    # df = df.dropna()
    # path_perfix = '/Users/Alchemist/Desktop/final_project/data/csv_files/'
    path_list = [['Writer_Cheng_train.csv', 'Writer_Cheng_test.csv'], 
        ['Writer_Peng_train.csv', 'Writer_Peng_test.csv'],
        ['Writer_Shen_train.csv', 'Writer_Shen_test.csv'],
        ['Writer_Wang_train.csv', 'Writer_Wang_test.csv']]
    single_path_list = [['task_train.csv', 'task_test.csv']]
    # train_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_train.csv'
    # test_path = '/Users/Alchemist/Desktop/final_project/data/csv_files/Writer_Shen_test.csv'
    params = edict({'epochs': 100, 'batchsize': 8, 'lr': 1e-4, 'num_classes': 10, 
        'data_root': '/gs/hs0/tga-shinoda/20M38216/final_project/data/dataset/',
        'csv_root': '/gs/hs0/tga-shinoda/20M38216/final_project/data/csv_files/'})

    solver = TrainSolver(params)

    # solver.lifelong_training(single_path_list)
    solver.train_base(single_path_list)

  

if __name__ == "__main__":
    
    main()








