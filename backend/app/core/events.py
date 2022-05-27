from typing import Callable
from backend.app.api.db.tasks import close_db_connection, connect_to_db

from fastapi import FastAPI


def preload_model():
    """
    In order to load model on memory to each worker
    """
    from services.predict import MachineLearningModelHandlerScore

    MachineLearningModelHandlerScore.get_model()


def create_start_app_handler(app: FastAPI) -> Callable:
    async def start_app() -> None:
        preload_model()
        await connect_to_db(app)

    return start_app

def create_stop_app_handler(app: FastAPI) -> Callable:
    async def start_app() -> None:
        await close_db_connection(app)

    return start_app
