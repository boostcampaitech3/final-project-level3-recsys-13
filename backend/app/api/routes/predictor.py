import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import *
from services.predict import MachineLearningModelHandlerScore as model

import re
import sqlalchemy
import numpy as np
import pandas as pd

from .postprocessing import filtering
from .globalVars import engine, user_reco_data_storage, interaction_modified, recipes, get_user_predictions, model_list, LABEL_CNT, update_batchpredict, ab


router = APIRouter()


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
    try:
        userid = data.user_id
        interacted = pd.read_sql(
            f"SELECT recipe_id FROM public.interactions WHERE user_id IN ({userid})", engine).recipe_id.values

        user_predictions = [ filtering(user_prediction, recipes, interacted, data.on_off_button, 
                                        data.ingredients_ls, data.max_sodium, data.max_sugar, data.max_minutes)
                            for user_prediction in get_user_predictions(userid) ]
        
        # 필터링 후 레시피가 10개 미만인 경우
        if sum(map(len, user_predictions)) <= LABEL_CNT: 
            top10_itemid, sources = [], []
            for i, user_prediction in enumerate(user_predictions):
                for item in user_prediction:
                    if item not in top10_itemid:
                        top10_itemid.append(item)
                        sources.append(i)
        else:
            top10_itemid, sources = blend_model_res(user_predictions, LABEL_CNT)
        
        # 유저별 추천 상황 기록 -> 일정 시간 지난 경우 삭제 구현 필요
        user_reco_data_storage[userid] = dict(zip(top10_itemid, sources))
        print(top10_itemid, sources)

        user_reco = []
        for id, name, url in recipes[recipes['id'].isin(top10_itemid)][['id', 'name', 'url']].values:
            user_reco.append({'id': id, 'name': name, 'url': url})

        return Top10RecipesResponse(lists=user_reco)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="서버 오류")


@router.get("/modcheck", description="interaction에 추가된 데이터가 있는지 검사합니다.")
async def return_answer():
    if interaction_modified:
        return GeneralResponse(state='True', detail='interaction modified')
    else:
        return GeneralResponse(state='False', detail='interaction not modified')


@router.post("/updatemodel", description="inference matrix를 업데이트된 정보로 변경합니다.")
async def return_answer(data: ModelUpdateRequest):
    update_batchpredict()
    global interaction_modified
    interaction_modified = False


