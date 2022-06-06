SHELL := /bin/bash

GCP_PROJECT_ID := stravasnooper-dev

# help:
#     @$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

deploy-dev:
	gcloud auth application-default login
	gcloud config set project stravasnooper-dev
	gcloud app deploy

run:
	python launch.py

format:
	black *.py

lint:
	pylint --disable=R,C,W1203,E1101 ./app

mlflow-launch:
# start cloud sql instance, has 10min timeout, takes ~15 to start up cloud sql, introduce wait to ensure it finishes
# if issues with cygwin make on windows not recognixing gcloud, see here: https://stackoverflow.com/questions/30749079/getting-gcloud-to-work-in-cygwin-windows
	echo Starting Cloud SQL instance, may take up to 15 min to be available
	gcloud sql instances patch stravasnooper-mlflow --activation-policy=ALWAYS

mlflow-shutdown:
# start cloud sql instance, has 10min timeout, takes ~15 to start up cloud sql, introduce wait to ensure it finishes
# if issues with cygwin make on windows not recognixing gcloud, see here: https://stackoverflow.com/questions/30749079/getting-gcloud-to-work-in-cygwin-windows
	echo Shutting down Cloud SQL instance, may take up to 15 min to be reflect in the Cloud Console
	gcloud sql instances patch stravasnooper-mlflow --activation-policy=NEVER