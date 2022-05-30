import os
import torch
import wandb
from args import parse_args

import foodcomTorch
import foodcomImplicit

from modeling_utils import setSeeds

def main(args):
    wandb.login()
    wandb.init(project="food_reco", config=vars(args))

    setSeeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.model == "als":
        train_data= foodcomImplicit.dataloader.load_data(args)
        foodcomImplicit.run(args, train_data)

    else:
        preprocess = foodcomTorch.Preprocess_interactions(args)
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()

        train_data, valid_data = preprocess.split_data(train_data)

        
        foodcomTorch.run(args, train_data, valid_data)

    

if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
