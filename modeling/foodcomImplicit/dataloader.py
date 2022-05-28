import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

def read_data(args):
    train_file_path = os.path.join(args.data_dir, "interactions_train.csv")
    valid_file_path = os.path.join(args.data_dir, "interactions_validation.csv")
    test_file_path = os.path.join(args.data_dir, "interactions_test.csv")

    df_train = pd.read_csv(train_file_path)
    df_valid = pd.read_csv(valid_file_path)
    df_test = pd.read_csv(test_file_path)

    if args.data_to_feed == "tr":
        pass
    elif args.data_to_feed == "trval":
        df_train = pd.concat([df_train, df_valid])
    elif args.data_to_feed == "all":
        df_train = pd.concat([df_train, df_valid, df_test])
    
    train_data = get_csr_matrix(df_train)
    valid_data = get_csr_matrix(df_valid)
    test_data = get_csr_matrix(df_test)


    return train_data, valid_data, test_data

def get_csr_matrix(df : pd.DataFrame, use_rating=False) -> csr_matrix:
    _view = [1]*df.shape[0] if not use_rating else df['rating'].values
    _matrix = csr_matrix((_view, (df['u'], df['i'])), \
                        shape=(df['u'].max()+1, df['i'].max()+1))
    return _matrix