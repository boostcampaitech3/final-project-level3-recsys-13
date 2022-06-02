#설치
install: 
	pip install pip --upgrade
	pip install 'apache-airflow==2.2.0'

server:
# cd airfow
	airflow db init
	export AIRFLOW_HOME=.
	airflow users create --username admin --password 1234 --firstname jinsu --lastname kim --role Admin --email aaa@naver.com
	airflow webserver --port=30005

scheduler:
	export AIRFLOW_HOME=.
	airflow scheduler