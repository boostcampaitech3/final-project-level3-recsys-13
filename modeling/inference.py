import warnings
import os
import pandas as pd
from datetime import datetime, timezone, timedelta

from google.cloud import storage
import sqlalchemy

import torch
import foodcomImplicit
import foodcomTorch

from args import parse_args
from dataloader import load_data, get_db_engine
from utils import wandb_download, best_model_finder


def main(args):
    # basic settings
    warnings.filterwarnings('ignore')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./core/storage.json"
    storage_client = storage.Client()
    KST = timezone(timedelta(hours=9))
    engine = get_db_engine()
    
    # batch_tag update in db <<before inference>>
    if args.update_batch_tag:
        meta_data = pd.read_sql(f"select * from public.meta_data", engine)
        meta_data['batch_tag'] += 1
        update_meta_data(meta_data, engine)
        
        return
    
    # best_model update in db <<after inference>>
    if args.inference_info:
        run_df = wandb_download
        best_model_str = best_model_finder(run_df)
        
        meta_data = pd.read_sql(f"select * from public.meta_data", engine)
        meta_data['best_model'] = best_model_str
        update_meta_data(meta_data, engine)
        
        return


    # model inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    engine = get_db_engine()
    meta_data = pd.read_sql(f"select * from public.meta_data", engine)
    batch_tag = meta_data['batch_tag'].item()
    args.batch_tag = batch_tag
    
    if args.model == 'als':
        interactions_df, _ = load_data()
        matrix_csr = foodcomImplicit.dataset.get_csr_matrix_inference(interactions_df)
        if foodcomImplicit.trainer.run_inference(args, matrix_csr):
            print('als inference...')
            bucket = storage_client.bucket('foodcom_als_models')
            bucket.blob(f'user_factors_{args.batch_tag}.npy').upload_from_filename(os.path.join(args.als_dir, 'user_factors.npy'))
            bucket.blob(f'item_factors_{args.batch_tag}.npy').upload_from_filename(os.path.join(args.als_dir, 'item_factors.npy'))
        print('als inference done.')
            
    elif args.model == 'multivae':
        pass
    
        
    
def update_meta_data(df:pd.DataFrame, engine) -> None:
    df.to_sql(name='meta_data',
                con=engine,
                schema='public',
                if_exists='replace',
                index=False,
                dtype={
                    'user_count': sqlalchemy.types.INTEGER(),
                    'recipe_count': sqlalchemy.types.INTEGER(),
                    'interaction_count': sqlalchemy.types.INTEGER(),
                    'best_model': sqlalchemy.types.Text(),
                    'batch_tag': sqlalchemy.types.INTEGER()
                    }
                )
    return
        
    

if __name__ == "__main__":    
    args = parse_args()
    main(args)
