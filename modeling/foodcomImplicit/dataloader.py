import os
import scipy
import numpy as np
import pandas as pd

def read_data(args):
    train_file_path = os.path.join(args.data_dir, "interactions_train.csv")
    valid_file_path = os.path.join(args.data_dir, "interactions_validation.csv")
    test_file_path = os.path.join(args.data_dir, "interactions_test.csv")

    df_train = pd.read_csv(train_file_path)
    df_valid = pd.read_csv(valid_file_path)
    df_test = pd.read_csv(test_file_path)

    train_data = get_csr_matrix(df_train)
    valid_data = get_csr_matrix(df_valid)
    test_data = get_csr_matrix(df_test)

    return train_data, valid_data, test_data

def get_csr_matrix(df : pd.DataFrame, use_rating=False) -> scipy.sparse.csr_matrix:
    _view = [1]*df.shape[0] if not use_rating else df['rating'].values
    _matrix = scipy.sparse.csr_matrix((_view, (df['u'], df['i'])), \
                                        shape=(df['u'].max()+1, df['i'].max()+1))
    return _matrix