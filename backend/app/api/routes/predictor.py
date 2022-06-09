import os
import joblib
import ast
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import GeneralRequest, GeneralResponse, RecoRequest, Top10RecipesResponse, RateRequest, SignUpRequest, SignInRequest, SignInResponse, ModelUpdateRequest, NumThemes, ThemeSample, ThemeSamples
from services.predict import MachineLearningModelHandlerScore as model

import numpy as np
import pandas as pd
import re
import sqlalchemy
import time


from .postprocessing import filtering
from .globalVars import engine, interaction_modified, meta_data, recipes, batchpredicts, model_list, LABEL_CNT, theme, theme_titles, update_batchpredict


router = APIRouter()


def get_user_predictions(user_id: int):
    '''
    model_list에 저장된 모델들의 예측 결과를 유저별로 반환
    '''
    return [ batchpredict[user_id] for batchpredict in batchpredicts ]


def blend_model_res(meta_data: pd.DataFrame, user_predict: list, top_k: int=10) -> tuple(list):
    items, sources = [], []
    a1, a2, a3, b1, b2, b3 = ast.literal_eval(meta_data['inference_traffic'].item())
    while len(items) <= top_k:
        sampling_list = [np.random.beta(a1, b1), np.random.beta(a2, b2), np.random.beta(a3, b3)]
        best_model = np.argsort(sampling_list)[-1]
        while True:
            rec_item = user_predict[best_model].pop()
            if rec_item not in items:
                items.append(rec_item)
                sources.append(best_model)
                break
    return items, sources


@router.post("/recten", description="Top10 recipes를 요청합니다")
async def return_top10_recipes(data: RecoRequest):
    userid = data.userid
    interacted = pd.read_sql(
        f"SELECT recipe_id FROM public.interactions WHERE user_id IN ({userid})", engine).id.values
    user_predictions = [ filtering(user_prediction, recipes, interacted, data.on_off_button, 
                                    data.ingredients_ls, data.max_sodium, data.max_sugar, data.max_minutes) 
                        for user_prediction in get_user_predictions(userid) ]
    
    top10_itemid, sources = blend_model_res(meta_data, user_predictions, LABEL_CNT)

    user_reco = []
    for id, name, description in recipes[recipes['id'].isin(top10_itemid)][['id', 'name', 'description']].values:
        user_reco.append({'id': id, 'name': name, 'description': description})

    return Top10RecipesResponse(lists=user_reco)


@router.post("/score", description="유저가 레시피에 점수를 남깁니다")
async def return_answer(data: RateRequest):
    global interaction_modified
    user_data = pd.read_sql(
        f"select * from public.user_data where user_id = {data.user_id};", engine)
    full_user_data = pd.read_sql(f"select * from public.user_data;", engine)
    meta_data = pd.read_sql(f"select * from public.meta_data;", engine)
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
                name='interactions',
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
        meta_data['interaction_count'] += user_data['interaction_count'].item()
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
                'best_model': sqlalchemy.types.TEXT(),
                'batch_tag': sqlalchemy.types.INTEGER()
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
            name='interactions',
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
        meta_data['interaction_count'] += user_data['interaction_count'].item()
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
                'best_model': sqlalchemy.types.TEXT(),
                'batch_tag': sqlalchemy.types.INTEGER()
            }
        )
        user_data['interactions'] = user_data['interactions'] + \
            ' ' + str(int(data.recipe_id))
        user_data['scores'] = user_data['scores'] + \
            ' ' + str(int(data.rating))
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
            f"select * from public.user_data where  name = '{data.name}';", engine)
        if str(user_data['password'].item()) == data.password:
            return SignInResponse(
                state='Approved',
                user_id=str(user_data['user_id'].item()),
                name=str(user_data['name'].item()),
                interactions=user_data['interactions'].item().split(),
                scores=user_data['scores'].item().split(),
                interaction_count=int(user_data['interaction_count'].item()),
                is_cold=user_data['cold_start'].item()
            )
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
    update_batchpredict()


@router.get("/num_themes", description="inference matrix를 업데이트된 정보로 변경합니다.")
async def return_answer():
    return NumThemes(num=len(theme))


@router.get("/theme/{theme_id}", description="inference matrix를 업데이트된 정보로 변경합니다.")
async def return_answer(theme_id: int):
    rec_list = theme[theme_id]
    rec_sample = np.random.choice(rec_list, 5)
    responses = []
    for id in rec_sample:
        responses.append(ThemeSample(
            id=id, title=recipes[recipes['id'] == id]['name'].item(), image=''))
    return ThemeSamples(theme_title=theme_titles[theme_id], samples=responses)
