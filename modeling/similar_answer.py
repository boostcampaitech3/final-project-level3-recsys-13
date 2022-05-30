import os
import pandas as pd
import numpy as np
import pickle
import ast
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

from utils import load_raw_data

from sentence_transformers import SentenceTransformer
import torch

def main(args):
    KST = timezone(timedelta(hours=9))
    if args.save_pass == 'default':
        args.save_pass = os.path.join(os.getcwd(), 'data/eval/asset')
    
    # load data & preprocessing
    interactive_df, recipes_df = load_raw_data()
    recipes_df['dish_recipe'] = recipes_df['steps'].apply(lambda x : " ".join(ast.literal_eval(x)))
    
    # make save dir
    bert_architect = os.path.join(os.getcwd(), 'bertArchitect')
    if not os.path.exists(bert_architect):
        os.makedirs(bert_architect)
    
    # make recipe_dict for naming
    recipe_dict = {}
    for j,i in enumerate(recipes_df['name']):
        recipe_dict[j] = i
    
    # bert encoding
    print('embedding... This will take about 20~25 minutes.')
    print(f'(start time: {datetime.now(KST).strftime("%Y-%m-%d_%H:%M:%S")})')
    model = load_model(args.model)
    encoding_recipes = model.encode(recipes_df['dish_recipe'])
    now = datetime.now(KST).strftime('%Y-%m-%d_%H:%M:%S')
    with open(f'./bertArchitect/recipe_embedding_{now}.pickle', 'wb') as f:
        pickle.dump(encoding_recipes, f)
    print('embedding saved.')
    
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
    with open(f'./bertArchitect/data_similarity_{now}.pickle', 'wb') as f:
        pickle.dump(data_similarity, f)
    print('data similarity saved.')
    
    # make similar_answer dictionary
    print('make similar answer...')
    similar_answer = dict()
    similarity_list = data_similarity.groupby(args.col_name1)[args.col_name2].apply(list)
    for idx, value in zip(similarity_list.index.to_numpy(), similarity_list.values):
        similar_answer[idx] = value
    if not os.path.exists(args.save_pass):
        os.makedirs(args.save_pass)
    with open(os.path.join(args.save_pass, 'similar_answer.pickle'), 'wb') as f:
        pickle.dump(similar_answer, f)
        
    print('Done')
    
    
def large_cosine_similarity(
    data:np.array, 
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size:int = 2000,
    col_name1:str = 'recipes1',
    col_name2:str = 'recipes2',
    top_n:int = 10
    ) -> pd.DataFrame:
    
    data = torch.tensor(data, dtype=torch.float32).to(device)
    col1 = torch.tensor([])
    col2 = torch.tensor([])
    cos_values = torch.tensor([])
    for batch in range(data.shape[0] // batch_size + 1): 
        temp_col1 = torch.tensor([], device=torch.device(device))
        temp_col2 = torch.tensor([], device=torch.device(device))
        temp_cos_values = torch.tensor([], device=torch.device(device))
        
        start_idx = batch * batch_size
        if start_idx+batch_size >= data.shape[0]:
            end_idx = data.shape[0]
        else:
            end_idx = start_idx + batch_size
    
        for idx in tqdm(range(start_idx, end_idx), 
                        desc=f'batch [{batch + 1} / {data.shape[0] // batch_size + 1}]: ',
                        leave=False):
            # broadcasting
            idx_cos_sim = torch.nn.functional.cosine_similarity(data[idx], data, dim=1)
            idx_cos_sim_argsorted = torch.argsort(idx_cos_sim, descending=True)
            
            idx_data = torch.tensor(idx, device=torch.device(device)).unsqueeze(0)
            idx_temp_col1 = torch.tensor([], device=torch.device(device))
            for i in range(top_n):
                idx_temp_col1 = torch.cat([idx_temp_col1, idx_data])
            temp_col1 = torch.cat([temp_col1, idx_temp_col1])
            
            idx_temp_col2 = idx_cos_sim_argsorted[1:1+top_n]
            temp_col2 = torch.cat([temp_col2, idx_temp_col2])
            
            idx_temp_values = idx_cos_sim[idx_temp_col2]
            temp_cos_values = torch.cat([temp_cos_values, idx_temp_values])
            
        col1 = torch.cat([col1, temp_col1.to('cpu')])
        col2 = torch.cat([col2, temp_col2.to('cpu')])
        cos_values = torch.cat([cos_values, temp_cos_values.to('cpu')])
            
    data_similarity = pd.DataFrame(
        {col_name1: col1.numpy().astype('int'), 
         col_name2: col2.numpy().astype('int'), 
         'cosine_similarity': cos_values.numpy()}
        )
    return data_similarity
    

def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #parser.add_argument('--path', type=str, default='/opt/ml/final-project-level3-recsys-13/modeling/data', help='데이터가 위치한 경로')
    parser.add_argument('--model', type=str, default='bert-large-nli-stsb-mean-tokens', help='pretrained 모델')
    parser.add_argument('--top_n', type=int, default=25, help='cosine_similarity에서 가져올 개수')
    parser.add_argument('--batch', type=int, default=2000, help='cosine_similarity 계산시 batch')
    parser.add_argument('--col_name1', type=str, default='recipes1', help='첫 번째 col 이름')
    parser.add_argument('--col_name2', type=str, default='recipes2', help='두 번째 col 이름')
    parser.add_argument('--save_pass', type=str, default='default', help='similar_answer 저장위치')
    
    args = parser.parse_args()
    print(args)
    main(args)
