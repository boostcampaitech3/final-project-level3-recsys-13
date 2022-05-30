import pandas as pd
import numpy as np
import pickle
import os
import yaml

from utils import setSeeds, load_raw_data

def main(args) -> None:
    '''
    데이터를 train / test로 분리합니다.
    
    args:
    path(str): RAW 데이터가 담겨있는 path. (TODO: db 이용시 수정해야 함).
    n_all(int): test set으로 분리할 아이템 개수(n_seq를 포함).
    n_seq(int): test set으로 분리할 seq 데이터 개수(seq 데이터: 시간 순서상 뒤에 있는 데이터).

    save:
      - train_interactions(pd.DataFrame): 학습에 사용할 유저 iteraction 정보.
      - test_interactions(pd.DataFrame): test할 유저 목록 (유저ID만 담겨있는 데이터프레임).
      - recipes_df(pd.DataFrame): 레시피 정보.
      - answer(list): 정답 데이터. shape = list[user_n][n_all]. 레시피 id가 담겨있음.
    '''
    if args.path == 'default':
        args.path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    print('data split ...')
    # load data & split data
    raw_interactions, raw_recipes = load_raw_data()
    train_df, test_df = data_split(raw_interactions, raw_recipes, args.n_all, args.n_seq)
    
    # answer file & test_df
    answer = list(test_df.groupby('user_id')['recipe_id'].apply(list))
    test_df = test_df[['user_id']]
    test_df[['recipe_id']] = 0
    test_df.reset_index(drop=True, inplace=True)
    
    # save data
    print('saving ...')
    save_train(train_df, raw_recipes, args.path)
    save_test(test_df, answer, args.path)
    
    # fix config
    if args.no_tag:
        print('Done.')
        return 
    
    with open('./core/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['batch_tag'] += 1
    with open('./core/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print('Done.')
    
    
def save_train(train_df, train_recipes, path):
    path = os.path.join(path, 'train')
    if not os.path.exists(path):
        os.makedirs(path)
    train_df.to_csv(os.path.join(path, 'train_interactions.csv'), index=False)
    train_recipes.to_csv(os.path.join(path, 'train_recipes.csv'), index=False)


def save_test(test_df, answer, path):
    path = os.path.join(path, 'eval')
    if not os.path.exists(path):
        os.makedirs(path)
    test_df.to_csv(os.path.join(path, 'sample_submission.csv'), index=False)
    
    path = os.path.join(path, 'asset')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'answer.pickle'), 'wb') as f:
        pickle.dump(answer, f)
    

def data_split(raw_interactions, raw_recipes, n_all, n_seq):
    setSeeds()
    trains, tests = [], []
    for usr_id, tp in raw_interactions.groupby('user_id', as_index=False):  
        _n_all = min(tp.shape[0]//2, n_all)
        _n_seq = min(_n_all, n_seq)  
        _n_static = _n_all - _n_seq

        _idxs = np.random.permutation(tp.shape[0]-_n_seq)[:_n_static]  # 랜덤으로 추출 (데이터 총 개수 - seq로 뽑아낼 개수)[:seq로 뺼 것 뺴고]
        _mask = tp.index.isin(tp.index[_idxs])
        for i in range(_n_seq):
            _mask[-i-1] = True

        trains.append(tp[~_mask])
        tests.append(tp[_mask])
        
    train_df = pd.concat(trains)
    test_df = pd.concat(tests)
    
    return train_df, test_df




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='default', help='데이터가 위치한 경로')
    parser.add_argument('--n_all', type=int, default=3, help='test셋으로 분리할 총 데이터 수(seq 포함)')
    parser.add_argument('--n_seq', type=int, default=1, help='test셋으로 분리할 seq 데이터 수')
    parser.add_argument('--no_tag', action='store_true')
    
    args = parser.parse_args()
    print(args)
    main(args)
