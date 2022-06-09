import os
import ast

import sqlalchemy
import numpy as np
import pandas as pd

from google.cloud import storage
from core.config import DATABASE_URL, GOOGLE_APPLICATION_CREDENTIALS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

LABEL_CNT = 10
interaction_modified = False
user_reco_data_storage = {}

def get_db_engine() -> sqlalchemy.engine.Engine:
    '''Returns a connection and a metadata object'''
    engine = sqlalchemy.create_engine(DATABASE_URL, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta


engine = get_db_engine()

recipes = pd.read_sql("select * from public.recipes", engine)
meta_data = pd.read_sql(f"select * from public.meta_data", engine)
model_list = ast.literal_eval(meta_data['best_model'].item())
inf_traffic = ast.literal_eval(meta_data['inference_traffic'].item())
ab = {}
for i, model in enumerate(model_list):
    ab[model] = (inf_traffic[i], inf_traffic[i+len(model_list)])

storage_client = storage.Client()
bucket = storage_client.bucket('foodcom_als_model')


def filter_download(bucket: storage.Bucket):
    bucket.blob('theme.npy').download_to_filename(
        'theme.npy')
    bucket.blob('theme_title.npy').download_to_filename(
            'theme_title.npy')


def batchpredict_download(bucket: storage.Bucket):
    '''
    metadata에 저장된 모델의 예측 결과를 다운로드
    '''
    for model in model_list:
        bucket.blob(f'{model}.npy').download_to_filename(f'{model}.npy')


def theme_download(bucket: storage.Bucket):
    bucket.blob('theme.npy').download_to_filename(
        'theme.npy')
    bucket.blob('theme_title.npy').download_to_filename(
            'theme_title.npy')


def batchpredict_download(bucket: storage.Bucket):
    '''
    metadata에 저장된 모델의 예측 결과를 다운로드
    '''
    for model in model_list:
        bucket.blob(f'{model}.npy').download_to_filename(f'{model}.npy')


theme_download(bucket)
theme = np.load('theme.npy', allow_pickle=True).item()
theme_titles = np.load('theme_title.npy', allow_pickle=True).item()


batchpredict_download(bucket)
batchpredicts = [ np.load(f'{model}.npy', allow_pickle=True).item() for model in model_list ]


def update_batchpredict():
    batchpredict_download(bucket)
    global batchpredicts
    batchpredicts = [ np.load(f'{model}.npy', allow_pickle=True).item() for model in model_list ]

def get_user_predictions(user_id: int):
    '''
    model_list에 저장된 모델들의 예측 결과를 유저별로 반환
    '''
    return [ batchpredict[user_id] for batchpredict in batchpredicts ]