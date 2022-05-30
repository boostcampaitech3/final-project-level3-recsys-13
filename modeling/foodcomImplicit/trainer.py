import os
import gc
import csv
import pickle
import yaml
from tqdm import tqdm

import torch

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

import implicit
from utils import recall_at_k
from foodcomImplicit.recommender import als_inference_trainer


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    isGPU = False
    if args.device == "cuda":
        isGPU = True
        print('using GPU')
    else:
        os.popen("export OPENBLAS_NUM_THREADS=1")
    
    if args.model == "als":
        model = implicit.als.AlternatingLeastSquares(factors=args.factors, 
                regularization=args.regularization, iterations=100, 
                random_state=args.seed, use_gpu=isGPU)
        
    return model


def recommend(model, user_item_matrix, args):
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
    result = np.argpartition(als_model_df, -args.top_k).iloc[:, -args.top_k:]
    
    recommended_recipes = list()
    recipes_columns = als_model_df.columns
    for user in range(result.shape[0]):
        user_recipes = list()
        for recipe_idx in result.values[user]:
            user_recipes.append(recipes_columns[recipe_idx])
        recommended_recipes.append(user_recipes)
            
    return recommended_recipes
            

def run(args, train_data: csr_matrix, user_item_matrix: pd.DataFrame):  # train_df, user_ids, recipe_ids
    torch.cuda.empty_cache()
    gc.collect()

    print('training & testing...')
    model = get_model(args)
    best_recall = -np.inf
    best_epochs = -np.inf
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        print('# Epoch {}: '.format(epoch), end='')
        model.fit(train_data, show_progress=False)
        recommended_recipes = recommend(model, user_item_matrix, args)
        recall = recall_at_k(args.test_dir, recommended_recipes, args.top_k)
        print(" Recall@{}= {}".format(args.top_k, recall))
        
        if best_recall < recall:
            best_epochs = epoch
            best_recall = recall
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break
            
    del model, user_item_matrix, recommended_recipes
        
    # model save
    model, inference_matrix = als_inference_trainer(args, best_epochs)
    print('save...')
    if not os.path.exists(args.als_dir):
        os.makedirs(args.als_dir)
    with open(f'{args.als_dir}/als_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{args.als_dir}/inference_matrix.pickle', 'wb') as f:
        pickle.dump(inference_matrix, f)

    # exp save 
    with open('./core/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open('./experiments.csv', 'a')as f:
        wr = csv.writer(f)
        wr.writerow(['als', best_recall, {'factors':args.factors, 'regularization':args.regularization, 'epoch':best_epochs}, config['batch_tag']])
    
    print('Done.')
    
    



