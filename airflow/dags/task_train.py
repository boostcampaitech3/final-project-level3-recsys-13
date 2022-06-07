from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pendulum
from fastapi import FastAPI
from airflow.models.baseoperator import chain

app = FastAPI()

# inference, batch tag update (bash) / request : model 이름을 담은 request

# def func_train():
#     import sys,os
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd())))+"/modeling")
# import train
import requests
models = ["als","als","als","als"]
# 앞의 03-python-operator-with-context는 provide_context=True 옵션을 주고 Attribute에 접근
# 이 방식이 아닌 Airflow의 Template 방식을 사용. Jinja Temolate => Flask에서 자주 쓰는 템플릿
# Python에서는 Template랑 provide_context=True와 큰 차이를 못 느낄 수도 있으나, SQL Operator나 다른 오퍼레이터에선 유용하게 사용됨(템플릿)
# 쿼리문(WHERE절)에 Airflow의 execution_date를 인자로 넣고 실행
# Jinja Template : Airflow의 미리 정의된 템플릿. {{ ds }}, {{ ds_nodash }} 라고 정의
# Airflow Operator에 넘겨주면 실행 과정에서 템플릿 기반으로 값이 업데이트됨
tasks_train = []
local_tz = pendulum.timezone("Asia/Seoul")
config = ["n_epochs"]
default_args = {
    'owner' : 'jinsu',
    'depends_on_past' : False, # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정됨. False는 과거의 실행 결과 상관없이 매일 실행
    'start_date' : datetime(2022, 6, 1, tzinfo=local_tz),
    'retries' : 3, # 실패시 재시도 횟수
    "retry_delay" : timedelta(minutes = 5) #만약 실패시 5분 뒤 재실행
}

def confirm_data():
    #0 이면 not change 1이면 train
    print("confirm_data")
    change_branchs = ["not_change", "train_start"]
    state = requests.get("localhost:aaaa/modcheck").json()
    flag = 1 if state["state"]=="True" else 0
    return change_branchs[flag]

def not_change():
    print("notchange:notchange")

def train_start(**context):
    # Train하기
    # key = context["params"]["config"]
    tmp = requests.post(f"http://118.67.132.123:30005/modeling", json= {"name":context["params"]["name"], "config":{}}).json(),

# with 구문으로 DAG 정의
with DAG(
    dag_id='train_start',
    default_args=default_args,
    schedule_interval = '0 0 0/2 * *',
    tags=['boostcamp_ai_final']
)as dag:

    # task_confirm_data = BranchPythonOperator(
    #     task_id = "confirm_data",
    #     python_callable = confirm_data
    # )

    task_batch_tag = PythonOperator(
        task_id="batch_tag",
        python_callable = train_start,
        params={"name":"--update_batch_tag"},
        provide_context=True
    )

    # task_not_change = PythonOperator(
    #     task_id="not_change",
    #     python_callable = not_change
    # )

    # task_train_start = PythonOperator(
    #     task_id="train_start",
    #     python_callable = train_start,
    #     params={"name":"als"},
    #     provide_context=True
    # )

    task_complete = PythonOperator(
        task_id="complete",
        python_callable = train_start,
        params={"name":"--inference_info"},
        provide_context=True
    )
    # task_train_start
    # task_batch_tag>>task_train_start>>task_complete

    # task_confirm_data >> task_batch_tag >> [task_train_start, task_not_change] 
    # task_batch_tag
    # task_train_start >> task_complete
    for i in range(len(models)):
        tasks_train.append(PythonOperator(
            task_id=f"model{i+1}",
            python_callable = train_start,
            params={"name":models[i]},
            provide_context=True
        ))

    task_batch_tag >> tasks_train >> task_complete
    
    # 업데이트, best모델 트리거