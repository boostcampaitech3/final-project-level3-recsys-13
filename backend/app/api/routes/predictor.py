from typing import Any

import os
import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.prediction import HealthResponse, MachineLearningResponse, UseridRequest, Top10RecipesResponse, RateRequest
from services.predict import MachineLearningModelHandlerScore as model

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

router = APIRouter()

############ SAMPLE CODE #############

def get_prediction(data_input: Any) -> MachineLearningResponse:
    return MachineLearningResponse(model.predict(data_input, load_wrapper=joblib.load, method="predict_proba"))


@router.get("/predict", response_model=MachineLearningResponse, name="predict:get-data")
async def predict(data_input: Any = None):
    if not data_input:
        raise HTTPException(status_code=404, detail=f"'data_input' argument invalid!")
    try:
        prediction = get_prediction(data_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return MachineLearningResponse(prediction=prediction)


@router.get("/health", response_model=HealthResponse, name="health:get-data")
async def health():
    is_health = False
    try:
        get_prediction("lorem ipsum")
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")

#######################################

data_dir = "/opt/final-project-level3-recsys-13/modeling/data/"
df = pd.read_csv(os.path.join(data_dir, "RAW_recipes.csv"))

try:
    with open(os.path.join(data_dir, "i_item.pkl"), 'rb') as f:
        i_item = pickle.load(f)
    with open(os.path.join(data_dir, "id_u.pkl"), 'rb') as f:
        id_u = pickle.load(f)
    assert isinstance(i_item, dict)
    assert isinstance(id_u, dict)
    csr = sp.load_npz(os.path.join(data_dir, "csr.npz"))
except Exception as e:
    from .generateChanger import generate_changer
    generate_changer()
    with open(os.path.join(data_dir, "i_item.pkl"), 'rb') as f:
        i_item = pickle.load(f)
    with open(os.path.join(data_dir, "id_u.pkl"), 'rb') as f:
        id_u = pickle.load(f)
    # assert isinstance(i_item, dict)
    # assert isinstance(id_u, dict)
    csr = sp.load_npz(os.path.join(data_dir, "csr.npz"))

user_factors: np.ndarray = np.load("/opt/final-project-level3-recsys-13/modeling/_als_userfactors.npy")
item_factors: np.ndarray = np.load("/opt/final-project-level3-recsys-13/modeling/_als_itemfactors.npy")

LABEL_CNT = 10


@router.post("/login", description="Top10 recipes를 요청합니다")
async def return_top10_recipes(data: UseridRequest):
    userid = data.userid
    useridx = id_u[userid]

    users_preference: np.ndarray = (user_factors[useridx] @ item_factors.T)
    users_preference[csr[useridx].nonzero()[1]] = float('-inf')
    top10_i = users_preference.argpartition(-LABEL_CNT)[-LABEL_CNT:]
    top10_itemid = [ i_item[i] for i in top10_i if i in i_item ]

    user_reco = []
    for id, name, description in df[df['id'].isin(top10_itemid)][['id','name','description']].values:
        user_reco.append( {'id': id, 'name': name, 'description': description} )

    return Top10RecipesResponse(lists = user_reco)


@router.post("/recommend", description="추천아이템을 요청합니다")
async def return_answer(data: RateRequest):
    if data.rate>5.0 or data.rate<0.0:
        return "잘못된 평점입니다. 0~5점 사이로 입력해주세요!"
    #TODO

    return data.rate
