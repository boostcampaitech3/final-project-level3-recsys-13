from pydantic import BaseModel
from typing import Optional


class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool


class UseridRequest(BaseModel):
    # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # default_factory : Product Class가 처음 만들어 질 때 호출되는 함수를 list로 하겠다 => Product 클래스를 생성하면 list를 만들어서 저장
    # uuid : 고유 식별자, Universally Unique Identifier
    userid: int
    # description: Optional[str] = None


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


class GeneralResponse(BaseModel):
    state: str
    detail: str
