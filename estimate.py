import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import pandas as pd

from elastic_weight_consolidation import ElasticWeightConsolidation
from dataset import NumeralDataset




class EstimateSolver(object):
    """docstring for EstimateSolver"""
    def __init__(self, csv_path, model_perfix, model_path_list, task_list, data_root, batch_size=8, lr =1e-5):
        super(EstimateSolver, self).__init__()
        self.data_root = data_root
        self.batch_size =batch_size
        df = pd.read_csv(csv_path)
        self.test_loaders = []
        for task in task_list:
            self.test_loaders.append(self.load_data(df[df['Writer'] == task]))

        self.loss_func = nn.CrossEntropyLoss()  
        resnet = torchvision.models.resnet18(pretrained=True)
        net = ElasticWeightConsolidation(resnet, crit=self.loss_func, lr=lr)
        self.models = []
        for i in model_path_list:
            self.models.append(self.load_checkpoint(net, model_perfix+i))
        
    def load_checkpoint(self, net, checkpoint_PATH):
        if checkpoint_PATH != None:
            model_CKPT = torch.load(checkpoint_PATH)
            net.model.load_state_dict(model_CKPT['state_dict'], strict=False)
            name = checkpoint_PATH.split('/')[-1]
            print(f'loading checkpoint {name}!')
            # import pdb
            # pdb.set_trace()
            return net
        

    def load_data(self, df):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = NumeralDataset(df, trans, self.data_root)
        data_loader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        return data_loader

    def evaluation(self):
        mtx = torch.zeros([len(self.models), len(self.test_loaders)], dtype=torch.float64)

        for net_idx, net in enumerate(self.models):
            for loader_idx, loader in enumerate(self.test_loaders):
                corr = 0
                for batchID, (img, label) in enumerate(loader):
                    output = net.model(img)
                    corr += (output.argmax(dim=1).long() == label).float().mean()

                acc = corr / len(loader)
                print(net_idx, loader_idx, acc)
                mtx[net_idx][loader_idx] = acc

        return mtx



def main():
    task_list = ['Shen', 'Wang', 'Peng', 'Cheng']
    model_perfix = '/gs/hs0/tga-shinoda/20M38216/final_project/lll_models/log11/'
    model_path_list = ['initial.pth', 'EWC_best_task_Shen.pth', 'EWC_best_task_Wang.pth',
        'EWC_best_task_Peng.pth', 'EWC_best_task_Cheng.pth']
    csv_path = '/gs/hs0/tga-shinoda/20M38216/final_project/data/csv_files/task_test.csv'
    data_root = '/gs/hs0/tga-shinoda/20M38216/final_project/data/dataset/'
    batch_size = 8

    solver = EstimateSolver(csv_path, model_perfix, model_path_list, task_list, data_root, batch_size)

    mtx = solver.evaluation()
    print(mtx)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()







