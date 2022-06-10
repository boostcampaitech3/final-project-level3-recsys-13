1. make
- 설치
    make install
- 서버 실행
    make server
- scheduler 실행
    make scheduler
- airflow/dags/core폴더에 .env파일을 넣어주세요

2. make 안될 시 
- airflow 설치
    pip install pip --upgrade
    pip install 'apache-airflow==2.2.0'

- 실행 위치
    cd airfow
    airflow db init

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