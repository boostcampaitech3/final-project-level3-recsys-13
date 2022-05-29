import os
import gc

import torch

from scipy.sparse import csr_matrix
import numpy as np

import implicit


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    isGPU = False
    if args.device == "cuda":
        isGPU = True
    else:
        os.popen("export OPENBLAS_NUM_THREADS=1")

    print('isGPU: ', isGPU)

    if args.model == "als":
        model = implicit.als.AlternatingLeastSquares(factors=args.factors,
                                                     regularization=args.regularization, iterations=2,
                                                     random_state=args.seed, use_gpu=isGPU)

    return model


def validate(model, valid_data, args) -> float:
    if isinstance(model.user_factors, np.ndarray):
        user_factors = model.user_factors
        item_factors = model.item_factors
    else:
        user_factors = model.user_factors.to_numpy()
        item_factors = model.item_factors.to_numpy()

    valid_cut = valid_data[:user_factors.shape[0], :item_factors.shape[0]]

    valid_data.nonzero()
    valid_prefrence = (user_factors[valid_cut.nonzero()[
                       0]] * item_factors[valid_cut.nonzero()[1]]).sum(axis=1)
    loss = np.abs(valid_cut.data-valid_prefrence).sum()/valid_data.data.size

    return loss


def run(args, train_data: csr_matrix):
    torch.cuda.empty_cache()
    gc.collect()

    model = get_model(args)
    # best_loss = float('inf')
    # early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        model.fit(train_data, show_progress=False)
        # cur_loss = validate(model, valid_data, args)
        print("Epoch: {}".format(epoch))
        # if cur_loss < best_loss:
        #     best_loss = cur_loss
        #     early_stopping_counter = 0
        #     if isinstance(model.user_factors, np.ndarray):
        #         user_factors = model.user_factors
        #         item_factors = model.item_factors
        #     else:
        #         user_factors = model.user_factors.to_numpy()
        #         item_factors = model.item_factors.to_numpy()
        #     np.save("_als_userfactors.npy", user_factors)
        #     np.save("_als_itemfactors.npy", item_factors)
        # else:
        #     early_stopping_counter += 1
        #     if early_stopping_counter >= args.patience:
        #         print(
        #             f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
        #         )
        #         break
