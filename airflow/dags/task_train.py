from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pendulum
# def func_train():
#     import sys,os
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd())))+"/modeling")
# import train
import requests


# 앞의 03-python-operator-with-context는 provide_context=True 옵션을 주고 Attribute에 접근
# 이 방식이 아닌 Airflow의 Template 방식을 사용. Jinja Temolate => Flask에서 자주 쓰는 템플릿
# Python에서는 Template랑 provide_context=True와 큰 차이를 못 느낄 수도 있으나, SQL Operator나 다른 오퍼레이터에선 유용하게 사용됨(템플릿)
# 쿼리문(WHERE절)에 Airflow의 execution_date를 인자로 넣고 실행
# Jinja Template : Airflow의 미리 정의된 템플릿. {{ ds }}, {{ ds_nodash }} 라고 정의
# Airflow Operator에 넘겨주면 실행 과정에서 템플릿 기반으로 값이 업데이트됨

local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner' : 'jinsu',
    'depends_on_past' : False, # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정됨. False는 과거의 실행 결과 상관없이 매일 실행
    'start_date' : datetime(2022, 5, 31, tzinfo=local_tz),
    'retries' : 3, # 실패시 재시도 횟수
    "retry_delay" : timedelta(minutes = 5) #만약 실패시 5분 뒤 재실행
}

def confirm_data():
    #0 이면 not change 1이면 train
    #flag = requests.post()
    print("confirm_data")
    flag = 1
    change_branchs = ["not_change", "train_start"]
    return change_branchs[flag]

def not_change():
    print("notchange:notchangeeeee")

def train():
    print("train:trainnnnnnn")

def train_modelA():
    print("trainA:AAAAAA")

def train_modelB():
    print("train_modelB:BBBBB")

def train_complete():
    #0 이면 not change 1이면 train
    #flag = requests.post()
    print("train_complete:trccccccc")
    flag = 1
    change_branchs = ["save_storage", "find_bestmodel"]
    return change_branchs[flag]

def save_storage():
    print("train_storage:tstststs")

def find_bestmodel():
    print("find_bestmodel:fbfbfbfb")
# with 구문으로 DAG 정의
with DAG(
    dag_id='train_start',
    default_args=default_args,
    schedule_interval = '0 0/2 * * *',
    tags=['boostcamp_ai_final']
)as dag:

    task_confirm_data = BranchPythonOperator(
        task_id = "confirm_data",
        python_callable = confirm_data
        
    )
    # import_task = BashOperator(
    #     task_id="command",
    #     bash_command = "echo hihi ; pwd",
    #     owner = "jinsu"
    # )
    task_not_change = PythonOperator(
        task_id="not_change",
        python_callable = not_change
    )

    task_train_start = PythonOperator(
        task_id='train_start',
        python_callable=train,
    )

    task_train_modelA = PythonOperator(
        task_id="train_modelA",
        python_callable=train_modelA
    )
    
    task_train_modelB = PythonOperator(
        task_id="train_modelB",
        python_callable=train_modelB
    )

    task_complete = BranchPythonOperator(
        task_id="complete",
        python_callable = train_complete
    )

    task_save_Storage = PythonOperator(
        task_id="save_storage",
        python_callable=save_storage
    )

    task_find_bestmodel = PythonOperator(
        task_id="find_bestmodel",
        python_callable=find_bestmodel
    )

    task_confirm_data >> [task_train_start, task_not_change] 
    task_train_start >> [task_train_modelA, task_train_modelB] >> task_complete >> [task_save_Storage, task_find_bestmodel]