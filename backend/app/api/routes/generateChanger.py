import pickle
import pandas as pd
import scipy.sparse as sp

def generate_changer():
    df_train = pd.read_csv("/opt/final-project-level3-recsys-13/modeling/data/interactions_train.csv")
    df_valid = pd.read_csv("/opt/final-project-level3-recsys-13/modeling/data/interactions_validation.csv")
    df_test = pd.read_csv("/opt/final-project-level3-recsys-13/modeling/data/interactions_test.csv")
    df_all = pd.concat([df_train, df_valid, df_test], axis=0)
    
    id_u = dict()
    for id, u in zip(df_all['user_id'], df_all['u']):
        id_u[id] = u
    with open('/opt/final-project-level3-recsys-13/modeling/data/id_u.pkl', 'wb') as f:
        pickle.dump(id_u, f, pickle.HIGHEST_PROTOCOL)

    i_item = dict()
    for item, i in zip(df_all['recipe_id'], df_all['i']):
        i_item[i] = item
    with open('/opt/final-project-level3-recsys-13/modeling/data/i_item.pkl', 'wb') as f:
        pickle.dump(i_item, f, pickle.HIGHEST_PROTOCOL)

    df_all['view'] = 1
    csr = sp.csr_matrix((df_all['view'], (df_all['u'], df_all['i'])), shape=(df_all['u'].max()+1, df_all['i'].max()+1))
    sp.save_npz("/opt/final-project-level3-recsys-13/modeling/data/csr.npz", csr)