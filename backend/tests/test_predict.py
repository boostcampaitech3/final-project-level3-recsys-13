import pytest
from app.api.routes.predictor import return_top10_recipes
from app.schema.prediction import UseridRequest

def test_return_top10_recipes():
    d = return_top10_recipes(UseridRequest(userid=2046))
    assert d.lists == []