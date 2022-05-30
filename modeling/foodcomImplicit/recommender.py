import os
import pickle

import numpy as np
import pandas as pd

import implicit
from utils import load_raw_data
from foodcomImplicit.dataloader import get_csr_matrix


# inference model
def als_inference_trainer(args, epoch:int):
    print('make inference model...')
    interactive_df, recipes_df = load_raw_data()
    matrix_csr, user_item_matrix = get_csr_matrix(interactive_df)
    
    isGPU = False
    if args.device == "cuda":
        isGPU = True
    else:
        os.popen("export OPENBLAS_NUM_THREADS=1")
    model = implicit.als.AlternatingLeastSquares(factors=args.factors, 
            regularization=args.regularization, iterations=epoch, 
            random_state=args.seed, use_gpu=isGPU)
    model.fit(matrix_csr, show_progress = False)
    
    inference_matrix = sort_inferece_matrix(model, user_item_matrix, args)
    
    return model, inference_matrix


def sort_inferece_matrix(model, user_item_matrix, args):
    if isinstance(model.user_factors, np.ndarray):
        user_factors = model.user_factors
        item_factors = model.item_factors
    else:
        user_factors = model.user_factors.to_numpy()
        item_factors = model.item_factors.to_numpy()
        
    als_model_df = pd.DataFrame(
        np.matmul(user_factors, 
                  item_factors.T), 
        index=user_item_matrix.index, columns=user_item_matrix.columns)
    als_model_df = als_model_df - user_item_matrix * 100  # masking
    result = np.argpartition(als_model_df, -args.inference_n).iloc[:, -args.inference_n:]

    return result


def recommend_inference(user_id, top_n):
    with open(f'./foodcomImplicit/architects/inference_matrix.pickle', 'rb') as f:
        inference_matrix = pickle.load(f)
    
    return({user_id: inference_matrix.iloc[user_id, :top_n].tolist()})

