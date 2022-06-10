import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import *

from pytz import timezone
from datetime import datetime
import sqlalchemy
import pandas as pd

from .globalVars import user_reco_data_storage, model_list, engine, interaction_modified


router = APIRouter()


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
    now = datetime.now(timezone('Asia/Seoul'))
    date = '%04d-%02d-%02d' % (now.year, now.month, now.day)
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
