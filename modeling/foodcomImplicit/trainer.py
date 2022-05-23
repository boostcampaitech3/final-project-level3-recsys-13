import gc
import torch

import scipy
import numpy as np

import implicit


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """

    if args.model == "als":
        model = implicit.als.AlternatingLeastSquares(factors=args.factors, 
                regularization=args.regularization, iterations=1, 
                random_state=args.seed, use_gpu=False)

    return model

def validate(model, valid_data, args) -> float:
    valid_cut = valid_data[:model.user_factors.shape[0], :model.item_factors.shape[0]]
    
    valid_data.nonzero()
    valid_prefrence = (model.user_factors[valid_cut.nonzero()[0]] * model.item_factors[valid_cut.nonzero()[1]]).sum(axis=1)
    loss = np.abs(valid_cut.data-valid_prefrence).sum()/valid_data.data.size


    return loss

def run(args, train_data : scipy.sparse.csr_matrix, valid_data : scipy.sparse.csr_matrix):
    torch.cuda.empty_cache()
    gc.collect()

    model = get_model(args)
    best_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        model.fit(train_data, show_progress=False)
        cur_loss = validate(model, valid_data, args)
        print("Epoch: {}, Loss: {}".format(epoch, cur_loss))
        if cur_loss < best_loss:
            best_loss = cur_loss
            early_stopping_counter = 0
            np.save("_als_userfactors.npy", model.user_factors)
            np.save("_als_itemfactors.npy", model.item_factors)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break