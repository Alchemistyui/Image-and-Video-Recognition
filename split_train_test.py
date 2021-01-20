import pandas as pd
import os
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

# path_prefix='/Users/Alchemist/Desktop/final_project/data/'
path_prefix='/gs/hs0/tga-shinoda/20M38216/final_project/data/dataset/'

# df = pd.read_csv(path_prefix+'dataset/task.csv')
# df = df.dropna()

# df['Class'] = df['FileName'].str.extract(r"([U]\d*)", expand=True)
# # class 1
# df[df['Class'] == 'U0030']

# cls_list = df['Class'].unique()

def split_task(key):
    tasks = df[key].unique()
    for task in tasks:
        df[df[key] == task].to_csv(path_prefix+'/csv_files/'+key+'_'+task+".csv", index=False)
        # print('write to '+path_prefix+key+'_'+task+".csv")


def split_train_test(out_dir, file_name, frac=0.8):
    file_path = out_dir+file_name
    df = pd.read_csv(file_path)
    skf = StratifiedKFold(n_splits=7, shuffle=True)

    train_idx, test_idx = list(skf.split(df, df['Class']))[0]
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    # train=df.sample(frac=frac)
    # test=df.drop(train.index)
    # import pdb
    # pdb.set_trace()
    train.to_csv(file_path.split('.')[0]+"_train.csv", index=False)
    test.to_csv(file_path.split('.')[0]+"_test.csv", index=False)
    # import pdb
    # pdb.set_trace()


def main():
    key = 'Writer'
    # split_task(key)
    split_train_test(path_prefix, 'task.csv')

    # for i in os.listdir(path_prefix+'/csv_files/'):
    #     if os.path.splitext(i)[1] == ".csv":
    #         split_train_test(path_prefix+'/csv_files/', i, 0.8)


if __name__ == "__main__":
    main()




