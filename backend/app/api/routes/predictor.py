from typing import Any, List

import os
import joblib
# from torch import R
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import GeneralRequest, GeneralResponse, UseridRequest, Top10RecipesResponse, RateRequest, SignUpRequest, SignInRequest, SignInResponse, ModelUpdateRequest, NumThemes, ThemeSample, ThemeSamples
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


def get_db_engine():
    '''Returns a connection and a metadata object'''
    engine = sqlalchemy.create_engine(DATABASE_URL, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


def ingredient_filter(recipes: pd.DataFrame, ingredients: list, use: bool):
    if use == True:
        l = "|".join(ingredients)
        filtered = recipes[~recipes.ingredients.str.contains(l)]
    elif use == "and":
        text = '^' + ''.join(fr'(?=.*{w})' for w in ingredients)
        filtered = recipes[~recipes['ingredients'].str.contains(text)]
    return filtered.id.values

# 칼로리 필터링


def calories_filter(recipes: pd.DataFrame, min_calories: float, max_calories: float):
    filtered = recipes[(recipes['calories'] < min_calories)
                       | (recipes['calories'] > max_calories)]
    return filtered.id.values

# 탄수화물 필터링


def carbohydrates_filter(recipes: pd.DataFrame, min_carbohydrates: float, max_carbohydrates: float):
    filtered = recipes[(recipes['carbohydrates (PDV)'] < min_carbohydrates) | (
        recipes['carbohydrates (PDV)'] > max_carbohydrates)]
    return filtered.id.values

# 단백질 필터링


def protein_filter(recipes: pd.DataFrame, min_protein: float, max_protein: float):
    filtered = recipes[(recipes['protein (PDV)'] < min_protein) | (
        recipes['protein (PDV)'] > max_protein)]
    return filtered.id.values

# 지방 필터링


def fat_filter(recipes: pd.DataFrame, min_fat: float, max_fat: float):
    filtered = recipes[(recipes['total fat (PDV)'] < min_fat)
                       | (recipes['total fat (PDV)'] > max_fat)]
    return filtered.id.values

# 포화지방 필터링


def saturated_fat_filter(recipes: pd.DataFrame, min_saturated_fat: float, max_saturated_fat: float):
    filtered = recipes[(recipes['saturated fat (PDV)'] < min_saturated_fat) | (
        recipes['saturated fat (PDV)'] > max_saturated_fat)]
    return filtered.id.values

# 나트륨 필터링


def sodium_filter(recipes: pd.DataFrame, min_sodium: float, max_sodium: float):
    filtered = recipes[(recipes['sodium (PDV)'] < min_sodium)
                       | (recipes['sodium (PDV)'] > max_sodium)]
    return filtered.id.values

# 당류 필터링


def sugar_filter(recipes: pd.DataFrame, min_sugar: float, max_sugar: float):
    filtered = recipes[(recipes['sugar (PDV)'] < min_sugar)
                       | (recipes['sugar (PDV)'] > max_sugar)]
    return filtered.id.values


# 연결
engine = get_db_engine()
interaction_modified = False
df = pd.read_sql("select * from public.recipes", engine)


LABEL_CNT = 10

storage_client = storage.Client()
bucket = storage_client.bucket('foodcom_als_model')
# bucket.blob('theme.npy').download_to_filename(
#         'theme.npy')
# bucket.blob('theme_title.npy').download_to_filename(
#         'theme_title.npy')


def model_download(item_dir, user_dir, bucket):
    bucket.blob(item_dir).download_to_filename(
        '_als_itemfactors.npy')
    bucket.blob(user_dir).download_to_filename(
        '_als_userfactors.npy')


model_download('_als_itemfactors.npy', '_als_userfactors.npy', bucket)
user_factors: np.ndarray = np.load("_als_userfactors.npy")
item_factors: np.ndarray = np.load("_als_itemfactors.npy")
theme = np.load('./theme.npy', allow_pickle=True).item()
theme_titles = np.load('./theme_title.npy', allow_pickle=True).item()
use_oven_recipe_ids = np.load('./use_oven_recipe_ids.npy')
LABEL_CNT = 10


@router.post("/recten", description="Top10 recipes를 요청합니다")
async def return_top10_recipes(data: UseridRequest):
    userid = data.userid
    ingredients = data.ingredients
    ingredient_use = data.ingredient_use
    button_oven, button_ingredients, button_calories, button_carbohydrates, button_protein, button_fat, button_saturated_fat, button_sodium, button_sugar = data.on_off_button
    min_calories, max_calories = data.calories
    min_carbohydrates, max_carbohydrates = data.carbohydrates
    min_protein, max_protein = data.protein
    min_fat, max_fat = data.fat
    min_saturated_fat, max_saturated_fat = data.saturated_fat
    min_sodium, max_sodium = data.sodium
    min_sugar, max_sugar = data.sugar
    user_preference: np.ndarray = (user_factors[userid] @ item_factors.T)
    interacted_recipes = pd.read_sql(
        f"SELECT recipe_id FROM public.interactions WHERE user_id IN ({userid})", engine)['recipe_id']
    total_filter = set()
    # 사용한 레시피
    interacted_recipes = [
        rid for rid in interacted_recipes if rid < user_preference.shape[0]]
    # 재료 필터
    if button_ingredients:
        ingredients_filtered_recipes = ingredient_filter(
            df, ingredients, ingredient_use)
        total_filter = total_filter | set(ingredients_filtered_recipes)
    # 칼로리 필터
    if button_calories:
        calories_filtered_recipes = calories_filter(
            df, min_calories, max_calories)
        total_filter = total_filter | set(calories_filtered_recipes)
    # 탄수화물 필터
    if button_carbohydrates:
        carbohydrates_filtered_recipes = carbohydrates_filter(
            df, min_carbohydrates, max_carbohydrates)
        total_filter = total_filter | set(carbohydrates_filtered_recipes)
    # 단백질 필터
    if button_protein:
        protein_filtered_recipes = calories_filter(
            df, min_protein, max_protein)
        total_filter = total_filter | set(protein_filtered_recipes)
    # 지방 필터
    if button_fat:
        fat_filtered_recipes = fat_filter(df, min_fat, max_fat)
        total_filter = total_filter | set(fat_filtered_recipes)
    # 포화지방 필터
    if button_saturated_fat:
        saturated_fat_filtered_recipes = saturated_fat_filter(
            df, min_saturated_fat, max_saturated_fat)
        total_filter = total_filter | set(saturated_fat_filtered_recipes)
    # 나트륨 필터
    if button_sodium:
        sodium_filtered_recipes = sodium_filter(df, min_sodium, max_sodium)
        total_filter = total_filter | set(sodium_filtered_recipes)
    # 당류 필터
    if button_sugar:
        sugar_filtered_recipes = sugar_filter(df, min_sugar, max_sugar)
        total_filter = total_filter | set(sugar_filtered_recipes)

    # 오븐 유무
    if not button_oven:
        total_filter = total_filter | set(use_oven_recipe_ids)

    user_preference[list(total_filter)] = float('-inf')
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
    model_download(data.item_factor, data.user_factor, bucket)


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
            id=id, title=df[df['id'] == id]['name'].item(), image=''))
    return ThemeSamples(theme_title=theme_titles[theme_id], samples=responses)
