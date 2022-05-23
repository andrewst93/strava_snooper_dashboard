SHELL := /bin/bash

GCP_PROJECT_ID := stravasnooper-dev

# help:
#     @$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

update-models:
# start cloud sql instance, has 10min timeout, takes ~15 to start up cloud sql, introduce wait to ensure it finishes
# if issues with cygwin make on windows not recognixing gcloud, see here: https://stackoverflow.com/questions/30749079/getting-gcloud-to-work-in-cygwin-windows
	gcloud sql instances patch stravasnooper-mlflow --activation-policy=ALWAYS
	sleep 300
	python -c 'from app import mlib; mlib.download_latest_production_models()'
	gcloud sql instances patch stravasnooper-mlflow --activation-policy=NEVER

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

format:
	black app/*.py

lint:
	pylint --disable=R,C,W1203,E1101 ./app

deploy:
	gcloud builds submit --region=us-central1 --config cloudbuild.yaml .

docker-build:
	docker build -t ssnoop-api:dev .

docker-run:
	echo Go to http://127.0.0.1:8080/ to connect to the API container.
	docker container run --name ssnoop-api-debug -e PORT=8080 -p 8080:8080 ssnoop-api:dev
# down:
#     docker-compose down