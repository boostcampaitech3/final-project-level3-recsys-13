import os
import random

import torch
import scipy
import pandas as pd
import numpy as np

class Preprocess_interactions():
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def __preprocessing(self, df, is_train=True):
        return df

    def __feature_engineering(self, df):
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__preprocessing(df, is_train)
        df = self.__feature_engineering(df)

        if is_train:
            self.args.num_users = df['u'].max() + 1
            self.args.num_items = df['i'].max() + 1

        return df

    def load_train_data(self, file_name):
        return None

    def load_test_data(self, file_name):
        return None


class InteractionsDataset(torch.utils.data.Dataset):
    '''
    Load Food.com dataset
    '''
    def __init__(self, data, args):
        self.data : scipy.sparse.csr_matrix = data
        self.args = args
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = InteractionsDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = InteractionsDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader