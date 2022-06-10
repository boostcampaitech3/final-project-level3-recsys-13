import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import *

import re
import sqlalchemy
import pandas as pd

from .globalVars import engine


router = APIRouter()


@router.post("/signup", description="회원가입을 요청합니다")
async def return_answer(data: SignUpRequest):
    if re.match('^[a-z0-9]+$', data.name) and re.match('^[a-z0-9]+$', data.password):
        names = set(pd.read_sql(
            "select name from public.user_data", engine)['name'])
        if data.name not in names:
            meta_data = pd.read_sql("select * from public.meta_data", engine)
            query = pd.DataFrame(
                {
                    'user_id': [int(meta_data['user_count'])],
                    'name': [data.name],
                    'password': [data.password],
                    'interaction_count': [0],
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
                    'interaction_count': sqlalchemy.types.INTEGER(),
                    'best_model': sqlalchemy.types.Text(),
                    'batch_tag': sqlalchemy.types.INTEGER(),
                    'inference_traffic': sqlalchemy.types.Text()
                }
            )
            return GeneralResponse(state='Approved', detail='Signup Success')
        else:
            return GeneralResponse(state='Denied', detail='duplicate error')
    else:
        return GeneralResponse(state='Denied', detail='format error')


@router.post("/signin", description="로그인을 요청합니다")
async def return_answer(data: SignInRequest):
    names = set(pd.read_sql(
        "select name from public.user_data", engine)['name'])
    if data.name in names:
        user_data = pd.read_sql(
            f"select * from public.user_data where name = '{data.name}';", engine)
        user_id = user_data['user_id'].item()
        interaction_data = pd.read_sql(
            f"select * from public.interactions where user_id = {user_id};", engine)
        interaction_list = [Interactions(recipe_id=interaction_data.iloc[i]['recipe_id'],
                                         score=interaction_data.iloc[i]['rating'],
                                         date=interaction_data.iloc[i]['date']) for i in range(interaction_data.shape[0])]
        log = list(interaction_data['recipe_id'])

        if str(user_data['password'].item()) == data.password:
            return SignInResponse(
                state='Approved',
                user_id=int(user_data['user_id'].item()),
                name=str(user_data['name'].item()),
                log=log,
                interactions=interaction_list,
                interaction_count=int(user_data['interaction_count'].item()),
                is_cold=user_data['cold_start'].item()
            )
        else:
            return GeneralResponse(state='Denied', detail='wrong password')
    else:
        return GeneralResponse(state='Denied', detail='undefined name')
