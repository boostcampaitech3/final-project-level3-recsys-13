import os
import random

import torch
import scipy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix

class Preprocess_interactions():
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, is_shuffle=True, seed=42):
        """
        split data into two parts with a given ratio.
        """
        if is_shuffle:
            data = shuffle(data, random_state=seed)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __preprocessing(self, df, is_train=True):
        return df

    def __feature_engineering(self, df):
        return df

    def load_data_from_file(self, is_train=True):
        train_file_path = os.path.join(self.args.data_dir, "interactions_train.csv")
        valid_file_path = os.path.join(self.args.data_dir, "interactions_validation.csv")
        test_file_path = os.path.join(self.args.data_dir, "interactions_test.csv")

        df_train = pd.read_csv(train_file_path)
        df_valid = pd.read_csv(valid_file_path)
        df_test = pd.read_csv(test_file_path)
        if self.args.data_to_feed == "tr":
            df = df_train
        elif self.args.data_to_feed == "trval":
            df = pd.concat([df_train, df_valid])
        elif self.args.data_to_feed == "all":
            df = pd.concat([df_train, df_valid, df_test])

        df = self.__preprocessing(df, is_train)
        df = self.__feature_engineering(df)

        if is_train:
            self.args.num_users = df['u'].max() + 1
            self.args.num_items = df['i'].max() + 1

        return df

    def load_train_data(self):
        self.train_data = self.load_data_from_file(self)

    def load_test_data(self, file_name):
        return None


class InteractionsDataset(torch.utils.data.Dataset):
    '''
    Load Food.com dataset
    '''
    def __init__(self, data, args):
        self.data = data
        self.args = args
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

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