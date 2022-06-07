from pydantic import BaseModel


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
    withoven: bool # 오븐 유무
    ingredients: list # 재료
    ingredient_use: bool # 선택한 재료 중 만들수 있는 레시피 True, 선택한 재료를 모두 사용하는 레시피 False
    calories: list # 칼로리 최소, 최대
    carbohydrates: list # 탄수화물 최소, 최대
    protein: list # 단백질 최소, 최대
    fat: list # 지방 최소, 최대



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
    is_cold: bool
    

class GeneralResponse(BaseModel):
    state: str
    detail: str


class NumThemes(BaseModel):
    num: int


class ThemeSamples(BaseModel):
    theme_title: str
    samples: list


class ThemeSample(BaseModel):
    id: int
    title: str
    image: str
