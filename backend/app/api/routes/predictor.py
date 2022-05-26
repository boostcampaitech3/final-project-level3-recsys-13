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

<<<<<<< HEAD
data_dir = "/opt/final-project-level3-recsys-13/modeling/data/"
df = pd.read_csv(os.path.join(data_dir, "RAW_recipes.csv"))

try:
    with open(os.path.join(data_dir, "i_item.pkl"), 'rb') as f:
        i_item = pickle.load(f)
    with open(os.path.join(data_dir, "id_u.pkl"), 'rb') as f:
        id_u = pickle.load(f)
    # assert isinstance(i_item, dict)
    # assert isinstance(id_u, dict)

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

=======
df = pd.read_csv("/opt/final-project-level3-recsys-13/modeling/data/RAW_recipes.csv")
with open('/opt/final-project-level3-recsys-13/modeling/data/i_item.pkl', 'rb') as f:
    i_item = pickle.load(f)
with open('/opt/final-project-level3-recsys-13/modeling/data/id_u.pkl', 'rb') as f:
    id_u = pickle.load(f)
assert isinstance(i_item, dict)
assert isinstance(id_u, dict)

csr = sp.load_npz('/opt/final-project-level3-recsys-13/modeling/data/csr.npz')
>>>>>>> origin/dev
user_factors: np.ndarray = np.load("/opt/final-project-level3-recsys-13/modeling/_als_userfactors.npy")
item_factors: np.ndarray = np.load("/opt/final-project-level3-recsys-13/modeling/_als_itemfactors.npy")

LABEL_CNT = 10

@router.post("/login", description="Top10 recipes를 요청합니다")
<<<<<<< HEAD
async def return_top10_recipes(data: UseridRequest):
=======
def return_top10_recipes(data: UseridRequest):
>>>>>>> origin/dev
    userid = data.userid
    userids = [ userid ]

    useridxs = [ id_u[userid] for userid in userids ]

    users_preferences = (user_factors[useridxs] @ item_factors.T)
    users_preferences[csr[useridxs, :].nonzero()] = float('-inf')
    top10s = [ m.argpartition(-LABEL_CNT)[-LABEL_CNT:] for m in users_preferences ]

    user_recos = []
    for top10 in top10s:
        ids = []
        for top in top10:
            try:
                ids.append(i_item[top])
            except KeyError:
                pass
        user_reco = []
        for id, name, description in df[df['id'].isin(ids)][['id','name','description']].values:
            user_reco.append({'id': id, 'name': name, 'description': description})
        user_recos.append(user_reco)
    return Top10RecipesResponse(lists = user_recos[0])


@router.post("/recommend", description="추천아이템을 요청합니다")
async def return_answer(data: RateRequest):
    if data.rate>5.0 or data.rate<0.0:
        return "잘못된 평점입니다. 0~5점 사이로 입력해주세요!"
    #TODO

    return data.rate
