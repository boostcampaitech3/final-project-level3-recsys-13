import os
import warnings

import torch
from args import parse_args
from foodcomImplicit.recommender import recommend_inference
from foodcomTorch import trainer
from utils import setSeeds, load_raw_data


def inference(model:str, user_id:int, top_n:int, seed:int = 42):
    '''
    als 추천 결과를 반환합니다
    
    parameters:
        - model:str = 모데명
        - user_id:int = 유저 id(한 명)
        - top_n:int = 몇 개의 아이템을 반환할 것인지
    '''
    # basic settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings('ignore')
    setSeeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if model == 'best':
        pass
    
    if model == 'als':
        return recommend_inference(user_id, top_n)
    
    elif model == 'multivae':
        pass

        
def als_inference(path:str = './foodcomImplicit/architects'):
    raw_interactions, _ = load_raw_data()
    

if __name__ == "__main__":
    print(inference(model='als', user_id=10, top_n=10))