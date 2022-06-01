from typing import Any

import os
import joblib
# from torch import R
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.prediction import GeneralRequest, GeneralResponse, UseridRequest, Top10RecipesResponse, RateRequest, SignUpRequest, SignInRequest, ModelUpdateRequest
from services.predict import MachineLearningModelHandlerScore as model

import numpy as np
import pandas as pd
import re
import sqlalchemy
import time
from core.config import DATABASE_URL, GOOGLE_APPLICATION_CREDENTIALS

from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

router = APIRouter()

############ SAMPLE CODE #############

# def get_prediction(data_input: Any) -> MachineLearningResponse:
#     return MachineLearningResponse(model.predict(data_input, load_wrapper=joblib.load, method="predict_proba"))


# @router.get("/predict", response_model=MachineLearningResponse, name="predict:get-data")
# async def predict(data_input: Any = None):
#     if not data_input:
#         raise HTTPException(status_code=404, detail=f"'data_input' argument invalid!")
#     try:
#         prediction = get_prediction(data_input)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Exception: {e}")

#     return MachineLearningResponse(prediction=prediction)


# @router.get("/health", response_model=HealthResponse, name="health:get-data")
# async def health():
#     is_health = False
#     try:
#         get_prediction("lorem ipsum")
#         is_health = True
#         return HealthResponse(status=is_health)
#     except Exception:
#         raise HTTPException(status_code=404, detail="Unhealthy")

#######################################

def get_db_engine():
    '''Returns a connection and a metadata object'''
    engine = sqlalchemy.create_engine(DATABASE_URL, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


# 연결
engine = get_db_engine()
interaction_modified = False
df = pd.read_sql("select * from public.recipes_df", engine)


LABEL_CNT = 10


def model_download(item_dir, user_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket('foodcom_als_models')
    bucket.blob(item_dir).download_to_filename(
        '_als_itemfactors.npy')
    bucket.blob(user_dir).download_to_filename(
        '_als_userfactors.npy')


model_download('_als_itemfactors.npy', '_als_userfactors.npy')
user_factors: np.ndarray = np.load("_als_userfactors.npy")
item_factors: np.ndarray = np.load("_als_itemfactors.npy")

LABEL_CNT = 10


@router.post("/recten", description="Top10 recipes를 요청합니다")
async def return_top10_recipes(data: UseridRequest):
    userid = data.userid

    user_preference: np.ndarray = (user_factors[userid] @ item_factors.T)
    interacted_recipes = pd.read_sql(
        f"SELECT recipe_id FROM public.interactions WHERE user_id IN ({userid})", engine)['recipe_id']
    interacted_recipes = [
        rid for rid in interacted_recipes if rid < user_preference.shape[0]]
    user_preference[interacted_recipes] = float('-inf')
    top10_itemid = user_preference.argpartition(-LABEL_CNT)[-LABEL_CNT:]

    # df를 메모리에 올릴지, 쿼리해서 줄지 고민. 현재는 메모리에 올림.
    user_reco = []
    for id, name, description in df[df['id'].isin(top10_itemid)][['id', 'name', 'description']].values:
        user_reco.append({'id': id, 'name': name, 'description': description})

    return Top10RecipesResponse(lists=user_reco)


@router.post("/score", description="유저가 레시피에 점수를 남깁니다")
async def return_answer(data: RateRequest):
    global interaction_modified
    user_data = pd.read_sql(
        f"select * from public.user_data where user_id = {data.user_id};", engine)
    full_user_data = pd.read_sql(f"select * from public.user_data;", engine)
    now = time.localtime()
    date = '%04d-%02d-%02d' % (now.tm_year, now.tm_mon, now.tm_mday)
    if user_data['cold_start'].item():
        if int(user_data['interaction_count'].item()) == 0:
            user_data['interactions'] = str(int(data.recipe_id))
            user_data['scores'] = str(int(data.rating))
        else:
            user_data['interactions'] = user_data['interactions'] + \
                ' ' + str(int(data.recipe_id))
            user_data['scores'] = user_data['scores'] + \
                ' ' + str(int(data.rating))
        user_data['interaction_count'] += 1
        if int(user_data['interaction_count'].item()) >= 10:
            user_data['interactions'] = 'None'
            user_data['scores'] = 'None'
            user_data['cold_start'] = False
            interaction = pd.DataFrame(
                {
                    'user_id': [RateRequest.user_id]*user_data['interaction_count'],
                    'recipe_id': user_data['interactions'].split(),
                    'date': [date],
                    'rating': user_data['scores'].split()
                }
            )
            interaction.to_sql(
                name='interactions_df',
                con=engine,
                schema='public',
                if_exists='append',
                index=False,
                dtype={
                    'user_id': sqlalchemy.types.INTEGER(),
                    'recipe_id': sqlalchemy.types.INTEGER(),
                    'date': sqlalchemy.types.TEXT(),
                    'rating': sqlalchemy.types.FLOAT(),
                }
            )
        user_data = user_data.squeeze()
        full_user_data[full_user_data['user_id']
                       == data.user_id] = user_data
        full_user_data.to_sql(
            name='user_data',
            con=engine,
            schema='public',
            if_exists='replace',
            index=False,
            dtype={
                'user_id': sqlalchemy.types.INTEGER(),
                'name': sqlalchemy.types.TEXT(),
                'password': sqlalchemy.types.TEXT(),
                'scores': sqlalchemy.types.TEXT(),
                'interaction_count': sqlalchemy.types.INTEGER(),
                'cluster': sqlalchemy.types.INTEGER(),
                'cold_start': sqlalchemy.types.BOOLEAN()
            }
        )
    else:
        interaction = pd.DataFrame(
            {
                'user_id': [data.user_id],
                'recipe_id': [data.recipe_id],
                'date': [date],
                'rating': [data.rating]
            }
        )
        interaction.to_sql(
            name='interactions_df',
            con=engine,
            schema='public',
            if_exists='append',
            index=False,
            dtype={
                'user_id': sqlalchemy.types.INTEGER(),
                'recipe_id': sqlalchemy.types.INTEGER(),
                'date': sqlalchemy.types.TEXT(),
                'rating': sqlalchemy.types.FLOAT(),
            }
        )
        user_data['interaction_count'] += 1
        user_data = user_data.squeeze()
        full_user_data[full_user_data['user_id']
                       == data.user_id] = user_data
        full_user_data.to_sql(
            name='user_data',
            con=engine,
            schema='public',
            if_exists='replace',
            index=False,
            dtype={
                'user_id': sqlalchemy.types.INTEGER(),
                'name': sqlalchemy.types.TEXT(),
                'password': sqlalchemy.types.TEXT(),
                'scores': sqlalchemy.types.TEXT(),
                'interaction_count': sqlalchemy.types.INTEGER(),
                'cluster': sqlalchemy.types.INTEGER(),
                'cold_start': sqlalchemy.types.BOOLEAN()
            }
        )
        interaction_modified = True
    return GeneralResponse(state='Approved', detail='Saved Interaction')


@router.post("/signup", description="회원가입을 요청합니다")
async def return_answer(data: SignUpRequest):
    if re.match('^[a-z0-9]+$', data.name) and re.match('^[a-z0-9]+$', data.password):
        names = set(pd.read_sql("select name from public.user_data", engine))
        if data.name not in names:
            meta_data = pd.read_sql("select * from public.meta_data", engine)
            query = pd.DataFrame(
                {
                    'user_id': [int(meta_data['user_count'])],
                    'name': [data.name],
                    'password': [data.password],
                    'interaction_count': [0],
                    'cluster': [0],
                    'cold_start': [True]
                }
            )
            query.to_sql(
                name='user_data',
                con=engine,
                schema='public',
                if_exists='append',
                index=False,
                dtype={
                    'user_id': sqlalchemy.types.INTEGER(),
                    'name': sqlalchemy.types.TEXT(),
                    'password': sqlalchemy.types.TEXT(),
                    'interaction_count': sqlalchemy.types.INTEGER(),
                    'cluster': sqlalchemy.types.INTEGER(),
                    'cold_start': sqlalchemy.types.BOOLEAN()
                }
            )
            meta_data['user_count'] += 1
            meta_data.to_sql(
                name='meta_data',
                con=engine,
                schema='public',
                if_exists='replace',
                index=False,
                dtype={
                    'user_count': sqlalchemy.types.INTEGER(),
                    'recipe_count': sqlalchemy.types.INTEGER(),
                    'interaction_count': sqlalchemy.types.INTEGER()
                }
            )
            return GeneralResponse(state='Approved', detail='Signup Success')
        else:
            return GeneralResponse(state='Denied', detail='duplicate error')
    else:
        return GeneralResponse(state='Denied', detail='format error')


@router.post("/signin", description="로그인을 요청합니다")
async def return_answer(data: SignInRequest):
    names = set(pd.read_sql("select name from public.user_data", engine))
    if data.name in names:
        user_data = pd.read_sql(
            f"select * from public.user_data where  name = '{data.name}';", engine)
        if str(user_data['password'].item()) == data.password:
            return GeneralResponse(state='Approved', detail='Signin Success')
        else:
            return GeneralResponse(state='Denied', detail='wrong password')
    else:
        return GeneralResponse(state='Denied', detail='undefined name')


@router.get("/modcheck", description="interaction에 추가된 데이터가 있는지 검사합니다.")
async def return_answer():
    if interaction_modified:
        return GeneralResponse(state='True', detail='interaction modified')
    else:
        return GeneralResponse(state='False', detail='interaction not modified')


@router.post("/updatemodel", description="inference matrix를 업데이트된 정보로 변경합니다.")
async def return_answer(data: ModelUpdateRequest):
    model_download(data.item_factor, data.user_factor)
