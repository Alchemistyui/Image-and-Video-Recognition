from sklearn.model_selection import StratifiedKFold






def training():
    # split_train_val(df, ratio)
    skf = StratifiedKFold(n_splits=7, shuffle=True)

    # 不需要变更dataset
    for fold, (train_idx, test_idx) in enumerate(skf.split(imgs, labels)):
        img_train, img_test = imgs.iloc[train_idx], imgs.iloc[test_idx]
        label_train, label_test = labels.iloc[train_idx], labels.iloc[test_idx]
            # print("TRAIN:", train_index, "TEST:", test_index)
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]

    train_loader = DataLoader(
        dataset=NumeralDataset(img_train, label_train),
        shuffle=True,
        batch_size=batchsize,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        dataset=NumeralDataset(img_test, label_test),
        shuffle=False,
        batch_size=batchsize,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )




def main():
    df = pd.read_csv('/Users/Alchemist/Desktop/2020-4q-assignment/Image_Video_Recognition/dataset/task.csv')
    df = df.dropna()

    

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)







