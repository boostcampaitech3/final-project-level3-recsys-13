import os
import pandas as pd
import numpy as np
import pickle
import ast
from datetime import datetime, timezone, timedelta

from dataloader import load_data
from utils import large_cosine_similarity
from sentence_transformers import SentenceTransformer

def bert2vec(args):
    KST = timezone(timedelta(hours=9))
    
    # load data & preprocessing
    _, recipes_df = load_data()
    recipes_df['dish_recipe'] = recipes_df['steps'].apply(lambda x : " ".join(ast.literal_eval(x)))
    
    # make save dir
    bert_architect = os.path.join(args.save_path, 'bertArchitect')
    if not os.path.exists(bert_architect):
        os.makedirs(bert_architect)
    
    # make recipe_dict for naming
    recipe_dict = {}
    for j,i in enumerate(recipes_df['name']):
        recipe_dict[j] = i
    
    # bert encoding
    print('embedding... This will take about 20~25 minutes.')
    print(f'(start time: {datetime.now(KST).strftime("%Y-%m-%d_%H:%M:%S")})')
    model = load_model(args.bert)
    encoding_recipes = model.encode(recipes_df['dish_recipe'])
    now = datetime.now(KST).strftime('%Y-%m-%d_%H:%M:%S')
    with open(f'{bert_architect}/recipe_embedding_{now}.pickle', 'wb') as f:
        pickle.dump(encoding_recipes, f)
    print('embedding saved.')
    
    if args.cal_similarity:
        # cal similarities
        print('cal similarities...')
        data_similarity = large_cosine_similarity(data = encoding_recipes,
                                                batch_size=args.batch,
                                                col_name1=args.col_name1,
                                                col_name2=args.col_name2,
                                                top_n = args.top_n)
        data_similarity[f'{args.col_name1}_name'] = data_similarity[args.col_name1].map(recipe_dict)
        data_similarity[f'{args.col_name2}_name'] = data_similarity[args.col_name2].map(recipe_dict)
        
        now = datetime.now(KST).strftime('%Y-%m-%d_%H:%M:%S')
        with open(f'{bert_architect}/data_similarity_{now}.pickle', 'wb') as f:
            pickle.dump(data_similarity, f)
        print('data similarity saved.')
    

def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model

