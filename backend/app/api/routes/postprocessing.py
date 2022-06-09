import pandas as pd
import numpy as np
from typing import List

'''
from postprocessing import filtering

### filtering 인자 설명 ###

user_recommendation_list -> 모델 추천 결과
recipes -> recipe dataframe
buttons -> length 7 리스트 형태
    0 : 재료필터 사용 버튼 (0-사용안함, 1-사용)
    1 : 나트륨 필터 (0-사용안함, 1-사용)
    2 : 당류 필터 (0-사용안함, 1-사용)
    3 : 조리시간 필터 (0-사용안함, 1-사용)
    4 : 탄수화물 필터 (0-사용안함, 1-저탄수, 2-고탄수)
    5 : 단백질 필터 (0-사용안함, 1-저단백, 2-고단백)
    6 : 지방 필터 (0-사용안함, 1-저지방, 2-고지방)
ingredients_ls -> 리스트 형태
max_sodium -> 나트륨 최대량
max_sugar -> 당류 최대량
max_minutes -> 최대 조리시간
filtering(a[0], recipes, [0, 0,0,0,0,0,0], ['sugar', 'onion'], 1000, 1000, 60)
'''


# 재료 필터
def ingredient_filter(recipes: pd.DataFrame, ingredients: list):
    text = '^' + ''.join(fr'(?=.*{w})' for w in ingredients)
    filtered = recipes[~recipes['ingredients'].str.contains(text)]
    return filtered.id.values

# 나트륨 필터링
def sodium_filter(recipes: pd.DataFrame, max_sodium: int):
    filtered = recipes[recipes['sodium (PDV)'] > max_sodium]
    return filtered.id.values
# 당류 필터링
def sugar_filter(recipes: pd.DataFrame, max_sugar: int):
    filtered = recipes[recipes['sugar (PDV)'] > max_sugar]
    return filtered.id.values

# 시간 필터링
def minutes_filter(recipes: pd.DataFrame, max_minutes: int):
    filtered = recipes[recipes['minutes'] > max_minutes]
    return filtered.id.values

# 탄단지 (안씀:0, 저:1 고:2)
def nutrition_rate(recipes: pd.DataFrame, button_carbohydrates, button_protein, button_fat):
    rate = np.array([0]*recipes.shape[0])

    if button_carbohydrates == 2:
        rate += np.array(pd.cut((recipes['carbohydrates (PDV)']/recipes['calories']), 5, labels=range(1,6)).values)
    elif button_carbohydrates == 1:
        rate += np.array(pd.cut((recipes['carbohydrates (PDV)']/recipes['calories']), 5, labels=list(reversed(range(1,6)))).values)

    if button_protein == 2:
        rate += np.array(pd.cut((recipes['protein (PDV)']/recipes['calories']), 5, labels=range(1,6)).values)
    elif button_protein == 1:
        rate += np.array(pd.cut((recipes['protein (PDV)']/recipes['calories']), 5, labels=list(reversed(range(1,6)))).values)

    if button_fat == 2:
        rate += np.array(pd.cut((recipes['total fat (PDV)']/recipes['calories']), 5, labels=range(1,6)).values)
    elif button_fat == 1:
        rate += np.array(pd.cut((recipes['total fat (PDV)']/recipes['calories']), 5, labels=list(reversed(range(1,6)))).values)
    
    return rate

# 필터링
def filtering(user_recommendation_list: list, recipes: pd.DataFrame, interacted: list,
                buttons: List[int], ingredients_ls: List[str], max_sodium: int, max_sugar: int, max_minutes: int):
    '''
    user_recommendation_list -> 모델 추천 결과
    recipes -> recipe dataframe
    buttons -> length 7 리스트 형태
        0 : 재료필터 사용 버튼 (0-사용안함, 1-사용)
        1 : 나트륨 필터 (0-사용안함, 1-사용)
        2 : 당류 필터 (0-사용안함, 1-사용)
        3 : 조리시간 필터 (0-사용안함, 1-사용)
        4 : 탄수화물 필터 (0-사용안함, 1-저탄수, 2-고탄수)
        5 : 단백질 필터 (0-사용안함, 1-저단백, 2-고단백)
        6 : 지방 필터 (0-사용안함, 1-저지방, 2-고지방)
    ingredients_ls -> 리스트 형태
    max_sodium -> 나트륨 최대량
    max_sugar -> 당류 최대량
    max_minutes -> 최대 조리시간
    filtering(a[0], recipes, [0,0,0,0,0,0,0], ['sugar', 'onion'], 1000, 1000, 60)
    '''
    button_ingredient, button_sodium, button_sugar, button_minutes, button_carbohydrates, button_protein, button_fat = buttons
    item2rate = dict((item, rank) for rank,item in enumerate(user_recommendation_list))
    filter_id = set()

    if interacted:
        filter_id = filter_id | set(interacted)
    if button_ingredient:
        filter_id = filter_id | set(ingredient_filter(recipes, ingredients_ls))
    if button_sodium:
        filter_id = filter_id | set(sodium_filter(recipes, max_sodium))
    if button_sugar:
        filter_id = filter_id | set(sugar_filter(recipes, max_sugar))
    if button_minutes:
        filter_id = filter_id | set(minutes_filter(recipes, max_minutes))

    filtered_df = recipes[recipes.id.isin(set(user_recommendation_list) - filter_id)]
    if filtered_df.shape[0] == 0:
        return []
    filtered_df['rank'] = filtered_df['id'].map(item2rate)
    filtered_df['nutrition_rank'] = nutrition_rate(filtered_df, button_carbohydrates, button_protein, button_fat)
    filtered_recommendation = filtered_df.sort_values(['n_ingredients','nutrition_rank','rank'], ascending=[True,False ,True]).id.values


    return filtered_recommendation