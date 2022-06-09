from pydantic import BaseModel
from typing import List


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool


class RecoRequest(BaseModel):
    # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # default_factory : Product Class가 처음 만들어 질 때 호출되는 함수를 list로 하겠다 => Product 클래스를 생성하면 list를 만들어서 저장
    # uuid : 고유 식별자, Universally Unique Identifier
    userid: int # 1

    # on_off_button
        # 0 : 재료필터 사용 여부    (0-사용안함, 1-사용)
        # 1 : 나트륨 필터           (0-사용안함, 1-사용)
        # 2 : 당류 필터             (0-사용안함, 1-사용)
        # 3 : 조리시간 필터         (0-사용안함, 1-사용)
        # 4 : 탄수화물 필터         (0-사용안함, 1-저탄수, 2-고탄수)
        # 5 : 단백질 필터           (0-사용안함, 1-저단백, 2-고단백)
        # 6 : 지방 필터             (0-사용안함, 1-저지방, 2-고지방)
    on_off_button: List[int]   # [0, 0, 0, 0, 0, 0, 0]

    # 필터링할 재료 이름
    ingredients_ls: list        # ['sugar', 'onion']

    max_sodium: int             # 1000
    max_sugar: int              # 1000
    max_minutes: int            # 60


class Top10RecipesResponse(BaseModel):
    lists: list


class RateRequest(BaseModel):
    user_id: int
    recipe_id: int
    rating: float


class SignInRequest(BaseModel):
    name: str
    password: str


class SignUpRequest(BaseModel):
    name: str
    password: str


class GeneralRequest(BaseModel):
    qeury: str
    detail: str


class ModelUpdateRequest(BaseModel):
    qeury: str
    user_factor: str
    item_factor: str


class SignInResponse(BaseModel):
    state: str
    user_id: str
    name: str
    interactions: List[int]
    scores: List[int]
    interaction_count: int
    is_cold: bool


class GeneralResponse(BaseModel):
    state: str
    detail: str


class NumThemes(BaseModel):
    num: int


class ThemeSample(BaseModel):
    id: int
    title: str
    image: str


class ThemeSamples(BaseModel):
    theme_title: str
    samples: List[ThemeSample]
