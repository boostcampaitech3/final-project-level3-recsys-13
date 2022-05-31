import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from dataloader import data_split, load_data

def read_data(args):
    interactions_df, _ = load_data()
    print('dataloading...')
    train_df, test_df = data_split(interactions_df, n_all=3, n_seq=1)
    train_data, user_item_matrix = get_csr_matrix(train_df)
    test_data = list(test_df.groupby('user_id')['recipe_id'].apply(list))
    return train_data, user_item_matrix, test_data

def get_csr_matrix(df : pd.DataFrame, use_rating=False) -> csr_matrix:
    df['viewed'] = np.ones(df.shape[0])
    user_item_matrix = df.pivot_table('viewed','user_id', 'recipe_id').fillna(0)
    matrix_csr = csr_matrix(user_item_matrix)
    
    return matrix_csr, user_item_matrix

def get_csr_matrix_inference(df: pd.DataFrame, use_rating=False) -> csr_matrix:
    view = [1]*df.shape[0] if not use_rating else df['rating'].values
    matrix_csr = csr_matrix((view, (df['user_id'], df['recipe_id'])),
                         shape=(df['user_id'].max()+1, df['recipe_id'].max()+1))
    return matrix_csr