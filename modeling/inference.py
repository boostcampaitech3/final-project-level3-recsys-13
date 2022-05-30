import warnings
import os
from datetime import datetime, timezone, timedelta

from google.cloud import storage

import torch
import foodcomImplicit
import foodcomTorch

from args import parse_args
from dataloader import load_data


def main(args):
    # basic settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings('ignore')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./core/storage.json"
    storage_client = storage.Client()
    KST = timezone(timedelta(hours=9))

    # model
    if args.model == 'best':
        pass
    
    if args.model == 'als':
        now = datetime.now(KST).strftime('%Y%m%d')
        
        interactions_df, _ = load_data()
        matrix_csr = foodcomImplicit.dataset.get_csr_matrix_inference(interactions_df)
        if foodcomImplicit.trainer.run_inference(args, matrix_csr):
            args.als_dir
            print('als inference...')
            bucket = storage_client.bucket('foodcom_als_models')
            bucket.blob(f'user_factors_{now}.npy').upload_from_filename(os.path.join(args.als_dir, 'user_factors.npy'))
            bucket.blob(f'item_factors_{now}.npy').upload_from_filename(os.path.join(args.als_dir, 'item_factors.npy'))
        print('als inference done.')
            
    
    elif args.model == 'multivae':
        pass
    
    

if __name__ == "__main__":    
    args = parse_args()
    main(args)
