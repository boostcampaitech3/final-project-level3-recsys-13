import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import *
from services.predict import MachineLearningModelHandlerScore as model

import re
import time
import sqlalchemy
import numpy as np
import pandas as pd

from .postprocessing import filtering
from .globalVars import engine, interaction_modified, recipes, get_user_predictions, model_list, LABEL_CNT, theme, theme_titles, update_batchpredict, ab


router = APIRouter()
user_reco_data_storage = {}


def blend_model_res(user_predict: list, top_k: int = 10):
    items, sources = [], []
    _all = pd.read_sql("SELECT COUNT(*) FROM public.model_interactions;", engine)['count'].item()
    _model = [ pd.read_sql(f"SELECT COUNT(*) FROM public.model_interactions WHERE model = '{model}';", engine)['count'].item() for model in model_list ]

    alp_bet = [ (ab[model][0]+_model[i], ab[model][1]+_all) for i, model in enumerate(model_list) ]
    while len(items) < top_k:
        sampling_list = [ np.random.beta(a, b) for (a, b) in alp_bet ]
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
    userid = data.user_id
    interacted = pd.read_sql(
        f"SELECT recipe_id FROM public.interactions WHERE user_id IN ({userid})", engine).recipe_id.values

    user_predictions = [ filtering(user_prediction, recipes, interacted, data.on_off_button, 
                                    data.ingredients_ls, data.max_sodium, data.max_sugar, data.max_minutes)
                        for user_prediction in get_user_predictions(userid) ]
    
    top10_itemid, sources = blend_model_res(user_predictions, LABEL_CNT)
    user_reco_data_storage[userid] = dict(zip(top10_itemid, sources))
    print(top10_itemid, sources)

    user_reco = []
    for id, name, description in recipes[recipes['id'].isin(top10_itemid)][['id', 'name', 'description']].values:
        user_reco.append({'id': id, 'name': name, 'description': description})

    return Top10RecipesResponse(lists=user_reco)


def check_Reco_Item(user_id: int, recipe_id: int, date: str):
    reco_model = None
    # 추천해준 목록에 있던 레시피인지 확인
    if user_id in user_reco_data_storage:
        reco_dict = user_reco_data_storage[user_id]
        if recipe_id in reco_dict:
            # 추천해준 목록에 있던 레시피이므로 모델의 점수를 업데이트
            model_id = reco_dict[recipe_id]
            reco_model = model_list[model_id]
            model_interaction = pd.DataFrame(
                {
                    'user_id': [user_id],
                    'recipe_id': [recipe_id],
                    'date': [date],
                    'model': [str(reco_model)],
                }
            )
            model_interaction.to_sql(
                name='model_interactions',
                con=engine,
                schema='public',
                if_exists='append',
                index=False,
                dtype={
                    'user_id': sqlalchemy.types.INTEGER(),
                    'recipe_id': sqlalchemy.types.INTEGER(),
                    'date': sqlalchemy.types.TEXT(),
                    'model': sqlalchemy.types.TEXT(),
                }
            )


@router.post("/score", description="유저가 레시피에 점수를 남깁니다")
async def return_answer(data: RateRequest):
    global interaction_modified
    user_data = pd.read_sql(
        f"select * from public.user_data where user_id = {data.user_id};", engine)
    full_user_data = pd.read_sql(f"select * from public.user_data;", engine)
    meta_data = pd.read_sql(f"select * from public.meta_data;", engine)
    now = time.localtime()
    date = '%04d-%02d-%02d' % (now.tm_year, now.tm_mon, now.tm_mday)
    check_Reco_Item(data.user_id, data.recipe_id, date)
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
    meta_data['interaction_count'] += 1
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
    user_data['interaction_count'] += 1
    if user_data['cold_start'].item() == True:
        if user_data['interaction_count'].item() >= 10:
            user_data['cold_start'] = False
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
            'interaction_count': sqlalchemy.types.INTEGER(),
            'cold_start': sqlalchemy.types.BOOLEAN()
        }
    )
    interaction_data = pd.read_sql(
        f"select * from public.interactions where user_id = {data.user_id};", engine)
    interaction_list = [Interactions(recipe_id=interaction_data.iloc[i]['recipe_id'],
                                     score=interaction_data.iloc[i]['rating'],
                                     date=interaction_data.iloc[i]['date']) for i in range(interaction_data.shape[0])]
    log = list(interaction_data['recipe_id'])
    interaction_modified = True
    return RateResponse(log=log, interactions=interaction_list, is_cold=user_data['cold_start'].item(), interaction_count=user_data['interaction_count'].item())


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


@router.get("/modcheck", description="interaction에 추가된 데이터가 있는지 검사합니다.")
async def return_answer():
    if interaction_modified:
        return GeneralResponse(state='True', detail='interaction modified')
    else:
        return GeneralResponse(state='False', detail='interaction not modified')


@router.post("/updatemodel", description="inference matrix를 업데이트된 정보로 변경합니다.")
async def return_answer(data: ModelUpdateRequest):
    update_batchpredict()


@router.get("/num_themes", description="전체 테마의 총량을 반환합니다.")
async def return_answer():
    return NumThemes(num=len(theme))


@router.get("/theme/{theme_id}", description="한개 테마를 받아 게시물을 반환합니다.")
async def return_answer(theme_id: int):
    rec_list = theme[theme_id]
    rec_sample = np.random.choice(rec_list, 5)
    responses = []
    for id in rec_sample:
        responses.append(ThemeSample(
            id=id, title=recipes[recipes['id'] == id]['name'].item(), image=recipes[recipes['id'] == id]['url'].item()))
    return ThemeSamples(theme_title=theme_titles[theme_id], theme_id=theme_id, samples=responses)


@router.post("/themes", description="테마 목록을 받아 게시물 리스트를 반환합니다.")
async def return_answer(data: ThemeListRequest):
    ret = []
    for theme_id in data.themes:
        rec_list = theme[theme_id]
        rec_sample = np.random.choice(rec_list, 5)
        responses = []
        for id in rec_sample:
            responses.append(ThemeSample(
                id=id, title=recipes[recipes['id'] == id]['name'].item(), image=recipes[recipes['id'] == id]['url'].item()))
        ret.append(ThemeSamples(
            theme_title=theme_titles[theme_id], theme_id=theme_id, samples=responses))
    return ThemeListResponse(articles=ret)


@router.get("/recipe/{recipe_id}", description="레시피 정보를 가져옵니다.")
async def return_answer(recipe_id: int):
    recipe_info = pd.read_sql(
        f"select * from public.recipes where public.recipes.id = {str(recipe_id)}", engine)

    def refine(sen):
        return(sen[1:])
    step = recipe_info["steps"].item()[1:-1].split("', ")
    step = list(map(refine, step))
    step[-1] = step[-1][:-1]
    return RecipeInfoResponse(
        name=recipe_info["name"].item(),
        id=recipe_info["id"].item(),
        minutes=recipe_info["minutes"].item(),
        submitted=recipe_info["submitted"].item(),
        tags=recipe_info["tags"].item(),
        nutrition=recipe_info["nutrition"].item(),
        steps=step,
        ingredients=recipe_info["ingredients"].item(),
        calories=recipe_info["calories"].item(),
        totalfat=recipe_info["total fat (PDV)"].item(),
        sugar=recipe_info["sugar (PDV)"].item(),
        sodium=recipe_info["sodium (PDV)"].item(),
        protein=recipe_info["protein (PDV)"].item(),
        saturatedFat=recipe_info["saturated fat (PDV)"].item(),
        carbohydrates=recipe_info["carbohydrates (PDV)"].item(),
        url=recipe_info["url"].item()
    )
