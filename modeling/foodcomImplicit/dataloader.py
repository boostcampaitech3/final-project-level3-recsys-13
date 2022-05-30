import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from utils import train_validation_split

def read_data(args):
    print('dataloading...')
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train_interactions.csv'))
    train_data, user_item_matrix = get_csr_matrix(train_df)

    return train_data, user_item_matrix

def get_csr_matrix(df : pd.DataFrame, use_rating=False) -> csr_matrix:
    df['viewed'] = np.ones(df.shape[0])
    user_item_matrix = df.pivot_table('viewed','user_id', 'recipe_id').fillna(0)
    matrix_csr = csr_matrix(user_item_matrix)
    
    return matrix_csr, user_item_matrix