# Kudos Prediction Model Deployment to GCP w/ FastAPI

To test the API locally you can launch it locally with the following command:

```
uvicorn app.main:app --reload
```

## Building Docker Image

To test the docker build locally use the following two commands.

```
docker build -t fastapiimage .
docker run -d --name mycontainer -p 8080:8080 fastapiimage

```

## Pushing Image to GCP for build

To build this API docker image and put into GCP container registry use the following command, see cloudbuild.yaml for details of where the container is stored.

```
gcloud builds submit --region us-central1 --config cloudbuild.yaml
```
