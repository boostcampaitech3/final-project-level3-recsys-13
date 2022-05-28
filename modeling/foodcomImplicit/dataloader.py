import os

import numpy as np
import pandas as pd
import sqlalchemy
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from scipy.sparse import csr_matrix


def connect(user, password, db, host='101.101.211.183', port=30003):
    '''Returns a connection and a metadata object'''
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    engine = sqlalchemy.create_engine(url, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


def read_data(args):
    train_file_path = os.path.join(args.data_dir, "interactions_train.csv")
    valid_file_path = os.path.join(
        args.data_dir, "interactions_validation.csv")
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


def load_data(args):
    engine = connect('admin', '1234', 'db')
    df = pd.read_sql("select * from public.interactions_df", engine)
    train_data = get_csr_matrix(df)
    return train_data


def get_csr_matrix(df: pd.DataFrame, use_rating=False) -> csr_matrix:
    _view = [1]*df.shape[0] if not use_rating else df['rating'].values
    _matrix = csr_matrix((_view, (df['user_id'], df['recipe_id'])),
                         shape=(df['user_id'].max()+1, df['recipe_id'].max()+1))
    return _matrix
