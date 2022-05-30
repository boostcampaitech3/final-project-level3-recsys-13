import pandas as pd
import numpy as np
import pickle
import os
import random
import torch

import sqlalchemy
from core.config import DATABASE_URL

def setSeeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def train_validation_split(df, n_valid:int = 1, n_seq:int = 1):
    trains, valids = [], []
    for usr_id, tp in df.groupby('user_id', as_index=False): 
        _n_all = min(tp.shape[0]//2, n_valid)
        _n_seq = min(_n_all, n_valid)  
        _n_static = _n_all - _n_seq

        _idxs = np.random.permutation(tp.shape[0]-_n_seq)[:_n_static]  # 랜덤으로 추출 (데이터 총 개수 - seq로 뽑아낼 개수)[:seq로 뺼 것 뺴고]
        _mask = tp.index.isin(tp.index[_idxs])
        for i in range(_n_seq):
            _mask[-i-1] = True

        trains.append(tp[~_mask])
        valids.append(tp[_mask])
        
    train_df = pd.concat(trains)
    valid_df = pd.concat(valids)
    return train_df, valid_df

        
def recall_at_k(test_path:str, predicted:list, topk:int) -> float:
    # actual
    with open(os.path.join(test_path, 'answer.pickle'), 'rb') as f:
        actual = pickle.load(f)
    
    # similar_answer
    with open(os.path.join(test_path, 'similar_answer.pickle'), 'rb') as f:
        similar_answer = pickle.load(f)
    
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        pred_set_with_similar = set()
        for j in pred_set:
            pred_set_with_similar.update(similar_answer[j])
            pred_set_with_similar.add(j)
        
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set_with_similar) / float(len(act_set))
            true_users += 1
            
    return sum_recall / true_users


def get_db_engine():
    '''Returns a connection and a metadata object'''
    engine = sqlalchemy.create_engine(DATABASE_URL, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


def load_raw_data():
    engine = get_db_engine()
    interactions_df = pd.read_sql("select * from public.interactions_df", engine)
    recipes_df = pd.read_sql("select * from public.recipes_df", engine)

    return interactions_df, recipes_df