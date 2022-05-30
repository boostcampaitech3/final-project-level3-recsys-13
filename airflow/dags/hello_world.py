from datetime import timedelta, datetime
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

def print_world():
    print("world")

with DAG(
    dag_id="hello_world",
    description = "My First DAG",
    start_date=datetime(2022,4,20),
    schedule_interval ="0 6 * * *",
    tags = ["my_dags"],
    
) as dag :

    t1 = BashOperator(
        task_id="print_hello",
        bash_command = "echo Hello",
        owner="heumsi",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t2 = PythonOperator(
        task_id="print_world",
        python_callable = print_world,
        depends_on_past=True,
        owner="heumsi",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t1 >> t2