import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from utils import AverageMeter
from elastic_weight_consolidation import ElasticWeightConsolidation
# from network import BaseModel
from dataset import NumeralDataset
# from config import BaseOptions

import pdb

class TrainSolver(object):
    """docstring for TrainSolver"""
    def __init__(self, params):
        super(TrainSolver, self).__init__()
        self.params = params
        self.loss_func = nn.CrossEntropyLoss()
        self.net = torchvision.models.resnet18(pretrained=True)
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr= self.params.lr)
        

        df = pd.read_csv(self.params.csv_root+self.params.csv_list[1])
        self.test_loaders = []
        for task in self.params.task_list:
            self.test_loaders.append(self.load_data(df[df['Writer'] == task]))


    def load_data(self, df, mode='test'):
        if mode == 'train':
            shuffle = True
        else:
            shuffle = False
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = NumeralDataset(df, trans, self.params.data_root)
        data_loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.params.batchsize,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        if mode == 'train':
            return data_loader, dataset
        else:
            return data_loader


    def train_base(self, task, train_df, test_df):
        # self.net = BaseModel(400*400, 100, 10)
        # train_df = pd.read_csv(self.params.csv_root+self.params.csv_list[0], 'train')
        # test_df = pd.read_csv(self.params.csv_root+self.params.csv_list[1])
        # self.load_data(csv_files[0])
        # self.load_data(train_df, test_df)
        writer = SummaryWriter(self.params.log_dir+'/'+'_'.join(self.params.task_list)+'/'+task+'/')
        train_loader, trainset = self.load_data(train_df, 'train')
        test_loader = self.load_data(test_df)

        self.loss_func = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=self.net.parameters(), lr= self.params.lr)

        # self.net.train()
        best_acc = 0
        for epo in range(self.params.epochs):
            self.net.train()
            losses = AverageMeter()
            for batchID, (img, label) in enumerate(train_loader):
                output = self.net(img)

                optim.zero_grad()
                loss = self.loss_func(output, label)
                loss.backward()
                optim.step()

                losses.update(loss.item(), img.size(0))
                # accs.update(corrects)
                del loss, output


            train_acc = self.eval(train_loader, 'train')
            test_acc, test_loss = self.eval(test_loader, 'need_loss')
            print(f'Epoch {epo:2d}: Train loss {losses.avg:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:4f}, Test loss {test_loss:4f}')
            writer.add_scalar('Train/Loss', losses.avg, epo)
            writer.add_scalar('Train/Acc', train_acc, epo)
            writer.add_scalar('Test/Acc', test_acc, epo)
            writer.add_scalar('Test/Loss', test_loss, epo)

            # remember the best acc.
            # if test_acc > best_acc:
            #     best_acc = test_acc
            #     torch.save(self.net.state_dict(), os.path.join(self.params.results_dir, 'Resnet_best_valid_'+str(round(test_acc.item(), 4))+'.pth'))


    def lifelong_training(self):
        train_df = pd.read_csv(self.params.csv_root+self.params.csv_list[0])
        test_df = pd.read_csv(self.params.csv_root+self.params.csv_list[1])
        # tasks = train_df[self.params.key].unique()

        mtx = torch.zeros([len(self.test_loaders)+1, len(self.test_loaders)], dtype=torch.float64)
        for loader_idx, loader in enumerate(self.test_loaders):
            test_acc = self.eval(loader, 'train')
            mtx[0][loader_idx] = test_acc


        for task_idx, task in enumerate(self.params.task_list):
            print('-'*80)
            print(task)
            task_train_df = train_df[train_df[self.params.key] == task]
            task_test_df = test_df[test_df[self.params.key] == task]
            self.train_base(task, task_train_df, task_test_df)
            for loader_idx, loader in enumerate(self.test_loaders):
                test_acc, _ = self.eval(loader, 'need_loss')
                mtx[task_idx+1][loader_idx] = test_acc

            print(mtx)

        return mtx



    def train(self, task, train_df, test_df):
        train_loader, trainset = self.load_data(train_df, 'train')
        test_loader = self.load_data(test_df)
        self.net.model.train()

        epoch_losses = AverageMeter()
        epoch_accs = AverageMeter()
        best_acc = 0
        writer = SummaryWriter(self.params.log_dir+'/'+'_'.join(self.params.task_list)+'/'+task+'/')
        for epo in range(self.params.epochs):
            # self.net.train()
            losses = AverageMeter()
            corrects = []
            # for batchID, (img, label) in tqdm(enumerate(self.train_loader)):
            for batchID, (img, label) in enumerate(train_loader):
                loss = self.net.forward_backward_update(img, label)
                losses.update(loss.item(), img.size(0))
                del loss

            train_acc = self.eval(train_loader, 'train')
            test_acc, test_loss = self.eval(test_loader, 'need_loss')
            print(f'Epoch {epo:2d}: Train loss {losses.avg:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:4f}, Test loss {test_loss:4f}')
            writer.add_scalar('Train/Loss', losses.avg, epo)
            writer.add_scalar('Train/Acc', train_acc, epo)
            writer.add_scalar('Test/Acc', test_acc, epo)
            writer.add_scalar('Test/Loss', test_loss, epo)
           

    def eval(self, dataloader, mode='need_loss'):
        # self.net.model.eval()
        self.net.eval()

        if mode != 'need_loss':
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


    def estimate_metrics(self, mtx):
        num_task = mtx.shape[1]
        accuracy = mtx[-1].sum() / num_task
        backward_transfer = 0
        for i in range(num_task):
            backward_transfer += (mtx[-1][i] - mtx[i+1][i])
        backward_transfer /= (num_task-1)
        forward_transfer = 0
        for i in range(1, num_task):
            forward_transfer += mtx[i][i] - mtx[0][i]
        forward_transfer /= (num_task-1)

        print('-'*80)
        print('Estimate Metrics')
        print(f'Accuracy {accuracy:4f}, Backward Transfer {backward_transfer:4f}, Forward Transfer {forward_transfer:4f}')



def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--key", type=str, default="Writer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--lr", type=int, default=1e-5)
    parser.add_argument("--num_classes", type=str, default=10)
    parser.add_argument("--data_root", type=str, default="/gs/hs0/tga-shinoda/20M38216/final_project/data/dataset/")
    parser.add_argument("--csv_root", type=str, default="/gs/hs0/tga-shinoda/20M38216/final_project/data/csv_files/")
    parser.add_argument("--log_dir", type=str, default="new_logs/")
    # self.parser.add_argument("--results_dir", type=str, default="/gs/hs0/tga-shinoda/20M38216/final_project/lll_models/log12/")
    parser.add_argument("--csv_list", default=['task_train.csv', 'task_test.csv'])
    parser.add_argument("--task_list", required=True, nargs='+')

    opt = parser.parse_args()

    args = vars(opt)
    for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

    return opt


def main():
    params = get_opt()


    solver = TrainSolver(params)
    
    mtx = solver.lifelong_training()
    solver.estimate_metrics(mtx)

  

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()








