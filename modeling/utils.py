import pandas as pd
import numpy as np
import pickle
import os
import random
import torch
import wandb
from tqdm import tqdm

def setSeeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def large_cosine_similarity(
    data:np.array, 
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size:int = 2000,
    col_name1:str = 'recipes1',
    col_name2:str = 'recipes2',
    top_n:int = 10
    ) -> pd.DataFrame:
    '''
    대용량 matrix에 대한 cosine similarity를 계산합니다.
    input data shape: (u*dim)
    output data shape: (u*top_n)
    '''
    
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


def recall_at_k(predicted:list, actual:list, topk:int) -> float:
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
            
    return sum_recall / true_users


       
def recall_at_k_with_similar_data(test_path:str, predicted:list, topk:int) -> float:
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


def wandb_download(runs:str = "boostcamp-relu/foodcom") -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(runs)

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame(summary_list)
    config_df = pd.DataFrame(config_list)
    name_df = pd.DataFrame(name_list)
    name_df.columns=['exp_name']
    
    run_df = pd.concat([name_df, config_df, summary_df], axis=1
                       )
    return run_df


def best_model_finder(run_df:pd.DataFrame) -> str:
    recent_batch_tag = run_df.batch_tag.max()
    recent_df = run_df[run_df['batch_tag'] == recent_batch_tag]
    
    best_score = recent_df['test recall'].max()
    best_model = recent_df[recent_df['test recall'] == best_score].iloc[0,:]
    
    best_model_str = f'{best_model["model"]}_{best_model["batch_tag"]}'
    
    return best_model_str
    