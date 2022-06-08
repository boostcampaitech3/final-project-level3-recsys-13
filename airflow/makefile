#설치
install: 
	pip install pip --upgrade
	pip install 'apache-airflow==2.2.0'

server:
# cd airfow
	export AIRFLOW_HOME=.
	airflow db init
	airflow users create --username admin --password 1234 --firstname jinsu --lastname kim --role Admin --email aaa@naver.com
	airflow webserver --port=30003

scheduler:
	export AIRFLOW_HOME=.
	airflow scheduler