from typing import Any

import os
import joblib
# from torch import R
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.prediction import GeneralResponse, UseridRequest, Top10RecipesResponse, RateRequest, SignUpRequest, SignInRequest
from services.predict import MachineLearningModelHandlerScore as model

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import re
import sqlalchemy
import time
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

def connect(user, password, db, host='101.101.211.183', port=30003):
    '''Returns a connection and a metadata object'''
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    engine = sqlalchemy.create_engine(url, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


# 연결
engine = connect('admin', '1234', 'db')

df = pd.read_sql("select * from public.recipes_df", engine)


# @router.post("/recten", description="Top10 recipes를 요청합니다")
# async def return_top10_recipes(data: UseridRequest):
#     userid = data.userid
#     userids = [userid]

#     useridxs = [id_u[userid] for userid in userids]

#     users_preferences = (user_factors[useridxs] @ item_factors.T)
#     users_preferences[csr[useridxs, :].nonzero()] = float('-inf')
#     top10s = [m.argpartition(-LABEL_CNT)[-LABEL_CNT:]
#               for m in users_preferences]

#     user_recos = []
#     for top10 in top10s:
#         ids = []
#         for top in top10:
#             try:
#                 ids.append(i_item[top])
#             except KeyError:
#                 pass
#         user_reco = []
#         for id, name, description in df[df['id'].isin(ids)][['id', 'name', 'description']].values:
#             user_reco.append(
#                 {'id': id, 'name': name, 'description': description})
#         user_recos.append(user_reco)
#     return Top10RecipesResponse(lists=user_recos[0])


@router.post("/score", description="유저가 레시피에 점수를 남깁니다")
async def return_answer(data: RateRequest):
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
    if re.match('^[a-z0-9]+$', data.name) and re.match('^[a-z0-9]+$', data.password):
        user_data = pd.read_sql(
            f"select * from public.user_data where  name = '{data.name}';", engine)
        if str(user_data['password'].item()) == data.password:
            return GeneralResponse(state='Approved', detail='Signin Success')
        else:
            return GeneralResponse(state='Denied', detail='wrong password')
    else:
        return GeneralResponse(state='Denied', detail='format error')
