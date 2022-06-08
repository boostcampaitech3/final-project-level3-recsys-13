import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

from recbole.quick_start import load_data_and_model

def main(args):
    path_dir = args.path
    file_list = os.listdir(path_dir)
    for file in file_list:
        if file.startswith(args.model_name):
            break
    
    result_dict = inference_top_n(os.path.join(path_dir, file))
    np.save(f'{path_dir}/{args.model_name}.npy', result_dict)
    

def inference_top_n(
    model_pth_path:str, 
    top_n:int = 100
    ) -> dict:
    
    _, model, dataset, _, _, test_data = load_data_and_model(model_pth_path)
    print('inference...')
    # device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token['user_id']
    item_id2token = dataset.field2id_token['recipe_id']

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    model.eval()
    for data in tqdm(test_data):
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)
        
        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction['user_id'].cpu().numpy()
        rating_pred[matrix[batch_user_index].toarray()[0] > 0] = -np.inf
        ind = np.argpartition(rating_pred, -top_n)[ -top_n:]
        sorted_ind_by_pred = ind[np.argsort(rating_pred[ind])[::-1]]
        
        # 예측값 저장
        if pred_list is None:
            pred_list = sorted_ind_by_pred
            user_list = np.array([batch_user_index.item() for _ in range(top_n)])
        else:
            pred_list = np.append(pred_list, sorted_ind_by_pred)
            user_list = np.append(user_list, np.array([batch_user_index.item() for _ in range(top_n)]))
        
    result = []
    for user, pred in zip(user_list, pred_list):
        result.append((int(user_id2token[user]), int(item_id2token[pred])))
            
    # 데이터 처리 : 딕셔너리로 반환
    result_df = pd.DataFrame(result, columns=["user", "item"])
    result_df.sort_values(by='user', inplace=True)
    result_groupby = result_df.groupby('user')['item'].apply(list)
    
    result_dict = dict()
    for user, item in zip(result_groupby.index.to_list(), result_groupby.values):
       result_dict[user] = item    
    
    print('inference done!')
    
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='BPR', type=str, help="model name")
    parser.add_argument("--path", default='/opt/ml/final-project-level3-recsys-13/modeling/RecBole/saved/0', type=str, help="saved path")
    
    args = parser.parse_args()
    
    main(args)