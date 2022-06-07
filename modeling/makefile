install:
	export OPENBLAS_NUM_THREADS=1
	conda config --append channels conda-forge
	conda install cudatoolkit=11.2 -c conda-forge --yes
	conda install cudatoolkit-dev=11.2 -c conda-forge --yes
	pip install --upgrade pip
	pip install poetry
	poetry install
	pip install psycopg2-binary
	pip install --upgrade google-cloud-storage

run_trigger:
	cd /opt/ml/final-project-level3-recsys-13/modeling
	python inference_trigger.py