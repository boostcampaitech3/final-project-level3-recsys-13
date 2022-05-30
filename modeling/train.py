import os
import torch
import wandb
from args import parse_args
import warnings
from datetime import datetime, timezone, timedelta

import foodcomTorch
from foodcomImplicit import dataloader, trainer

from utils import setSeeds

def main(args):
    #wandb.login()
    #wandb.init(project="food_reco", config=vars(args))

    # basic settings
    KST = timezone(timedelta(hours=9))
    warnings.filterwarnings('ignore')
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # model
    if args.model == 'all':
        pass
    
    elif args.model == "als":
        now = datetime.now(KST).strftime('%Y-%m-%d_%H:%M:%S')
        print(f'als ! [start time: {now}]')
        train_data, user_item_matrix = dataloader.read_data(args)
        trainer.run(args, train_data, user_item_matrix)
        
    elif args.model == 'torch-':
        pass
    
        # preprocess = foodcomTorch.Preprocess_interactions(args)
        # preprocess.load_train_data(args.file_name)
        # train_data = preprocess.get_train_data()
        # train_data, valid_data = preprocess.split_data(train_data)
        # foodcomTorch.run(args, train_data, valid_data)
        

    

if __name__ == "__main__":
    args = parse_args(mode="train")
    main(args)
