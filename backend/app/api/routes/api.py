from fastapi import APIRouter

from api.routes import predictor, recipes, users, themes

router = APIRouter()
router.include_router(predictor.router, tags=["predictor"], prefix="/v1")
router.include_router(recipes.router, tags=["recipes"], prefix="/v1")
router.include_router(users.router, tags=["users"], prefix="/v1")
router.include_router(themes.router, tags=["themes"], prefix="/v1")
