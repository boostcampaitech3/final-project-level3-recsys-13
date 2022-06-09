import joblib
from core.errors import PredictException
from fastapi import APIRouter, HTTPException
from loguru import logger
from schema.types import *

import numpy as np

from .globalVars import theme, recipes, theme_titles


router = APIRouter()


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