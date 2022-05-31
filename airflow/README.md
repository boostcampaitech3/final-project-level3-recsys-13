- airflow 설치
    pip install pip --upgrade
    pip install 'apache-airflow==2.2.0'

- 실행 위치
    cd airfow

    export AIRFLOW_HOME=.
    export PYTHONPATH=$PYTHONPATH:/opt/ml/~~~/final-project-level3-recsys-13/modeling/

    - 계정 생성
        airflow users create --username 아이디 --password 비밀번호 --firstname 이름 --lastname 성 --role Admin --email ~~~@~~~.com

    - 실행
        airflow webserver --port 30004

        다른 터미널열어서
            export AIRFLOW_HOME=.
            export PYTHONPATH=$PYTHONPATH:/opt/ml/~~~/final-project-level3-recsys-13/modeling/
            airflow scheduler