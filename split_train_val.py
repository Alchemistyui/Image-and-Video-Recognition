import pandas as pd
import os

path_prefix='/Users/Alchemist/Desktop/final_project/data/'

df = pd.read_csv(path_prefix+'dataset/task.csv')
df = df.dropna()

df['Class'] = df['FileName'].str.extract(r"([U]\d*)", expand=True)
# class 1
df[df['Class'] == 'U0030']

cls_list = df['Class'].unique()

def split_task(key):
    tasks = df[key].unique()
    for task in tasks:
        df[df[key] == task].to_csv(path_prefix+'/csv_files/'+key+'_'+task+".csv", index=False)
        # print('write to '+path_prefix+key+'_'+task+".csv")


def split_train_test(out_dir, file_name, frac):
    file_path = path_prefix+file_name
    df = pd.read_csv(file_path)
    train=df.sample(frac=frac)
    test=df.drop(train.index)
    train.to_csv(file_path.split('.')[0]+"_train.csv", index=False)
    test.to_csv(file_path.split('.')[0]+"_test.csv", index=False)


def main():
    key = 'Writer'
    # split_task(key)

    for i in os.listdir(path_prefix):
        if os.path.splitext(i)[1] == ".csv":
            split_train_test(path_prefix+'/csv_files/', i, 0.8)


if __name__ == "__main__":
    main()