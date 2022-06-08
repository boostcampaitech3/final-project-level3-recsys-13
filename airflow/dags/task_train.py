from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pendulum
from airflow.models.baseoperator import chain
import yaml
import sqlalchemy
from core.config import DATABASE_URL
import pandas as pd
import requests


models = ["BPR", "LightGCN", "MultiVAE", "MultiDAE", "CDAE", "RecVAE"]

def get_db_engine():
    '''Returns a connection and a metadata object'''
    engine = sqlalchemy.create_engine(DATABASE_URL, echo=True)
    #meta = sqlalchemy.MetaData(bind=engine, reflect=True)
    return engine  # , meta

tasks_train = []
#local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner' : 'jinsu',
    'depends_on_past' : False, # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정됨. False는 과거의 실행 결과 상관없이 매일 실행
    #'start_date' : datetime(2022, 6, 7, tzinfo=local_tz),
    'start_date' : datetime(2022, 6, 7),
    'retries' : 3, # 실패시 재시도 횟수
    "retry_delay" : timedelta(minutes = 1) #만약 실패시 1분 뒤 재실행
}

def confirm_data():
    #0 이면 not change 1이면 train
    print("confirm_data")
    change_branchs = ["not_change", "batch_tag"]
    
    with open('info.yaml') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    
    engine = get_db_engine()
    meta = pd.read_sql(f"select * from public.meta_data", engine)
    db_user_n = meta['user_count'].item()

    flag = 1 if info["now_user_n"] < db_user_n else 0
    
    info['now_user_n'] = db_user_n
    with open('info.yaml', 'w') as f:
        yaml.dump(info, f)
        
    return change_branchs[flag]

def not_change():
    print("notchange:notchange")

def train_start(**context):
    # Train하기
    tmp = requests.post(f"http://118.67.132.123:30001/modeling", json= {"name":context["params"]["name"], "config":{}}).json(),

# with 구문으로 DAG 정의
with DAG(
    dag_id='train_start',
    default_args=default_args,
    schedule_interval = '0 0 0/4 * *',
    tags=['boostcamp_ai_final']
)as dag:

    task_confirm_data = BranchPythonOperator(
        task_id = "confirm_data",
        python_callable = confirm_data
    )

    task_batch_tag = PythonOperator(
        task_id="batch_tag",
        python_callable = train_start,
        params={"name":"--update_batch_tag"},
        provide_context=True
    )

    task_not_change = PythonOperator(
        task_id="not_change",
        python_callable = not_change
    )

    task_complete = PythonOperator(
        task_id="complete",
        python_callable = train_start,
        params={"name":"--inference_info"},
        provide_context=True
    )

    for i in range(len(models)):
        tasks_train.append(PythonOperator(
            task_id=f"model{i+1}",
            python_callable = train_start,
            params={"name":models[i]},
            provide_context=True
        ))

    task_confirm_data >> [task_batch_tag, task_not_change]
    task_batch_tag >> tasks_train >> task_complete
    