import uvicorn
from api.routes.api import router as api_router
from core.config import API_PREFIX, DEBUG, PROJECT_NAME, VERSION
from core.events import create_start_app_handler
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException


def get_application() -> FastAPI:
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    pre_load = False
    if pre_load:
        application.add_event_handler("startup", create_start_app_handler(application))
    return application


app = get_application()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, debug=False)


# from typing import Optional
# from fastapi import FastAPI, APIRouter
# from fastapi.param_functions import Depends
# import uvicorn
# import pandas as pd
# from pydantic import BaseModel, Field
# from typing import List, Optional, Union
# from uuid import UUID, uuid4

# import pandas as pd
# import numpy as np

# df = pd.read_csv("RAW_recipes.csv")
# # df_router = APIRouter(prefix="/df")
# # answer_router = APIRouter(prefix="/answer")

# app = FastAPI()

# answercode = []

# class Recipes(BaseModel):
#     # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
#     # default_factory : Product Class가 처음 만들어 질 때 호출되는 함수를 list로 하겠다 => Product 클래스를 생성하면 list를 만들어서 저장
#     # uuid : 고유 식별자, Universally Unique Identifier
#     login : str
#     # description: Optional[str] = None

# class Rate(BaseModel):
#     rate : float

# class Response(BaseModel):
#     lists : list


# @app.on_event("startup")
# def startup_event():
#     print("Application startup")

# @app.on_event("shutdown")
# def shutdown_event():
#     print("Shutdown Event!")


# @app.get("/answer", description="주문 리스트를 가져옵니다")
# async def get_orders():
#     return answercode

# @app.get("/answer/{answer_id}", description="Order 정보를 가져옵니다")
# async def get_order(answer_id: int) -> float:
#     if answer_id>=len(answercode):
#         return {"message": "주문 정보를 찾을 수 없습니다"}
#     return answercode[answer_id]

# @app.post("/login", description="Top10 recipes를 요청합니다")
# def return_answer(data: Recipes):
#     #TODO
#     idx = np.random.randint(0,len(df)-1, 10)
#     dd = []
#     for i in idx:
#         tmp = df.iloc[i]        
#         while type(tmp.description)!=str:
#             i=(i+1)%len(df)
#             tmp = df.iloc[i]
#         dd.append({"id": int(tmp["id"]), "name": tmp["name"], "description": tmp.description})
#     return Response(lists = dd)


# @app.post("/recommend", description="추천아이템을 요청합니다")
# async def return_answer(rate: Rate):
#     if rate.rate>5.0 or rate.rate<0.0:
#         return "잘못된 평점입니다. 0~5점 사이로 입력해주세요!"
#     #TODO

#     return rate.rate


# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)
