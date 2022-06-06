# Strava Snooper Dashboard

Digging deeper into personal Strava data to figure out how training has changed over the years and seeing what insights Strava might be able to gleen from all our data.

# Requirements

The primary dashboard and model development/training is encompassed in the requirements.txt for all development and tools for deployment.

To ensure the requirements.txt is updated in the format expected by App Engine use the following command to generate one in the correct format from a conda environment:

`pip list --format=freeze > requirements.txt`

Packages to be removed from requirements for deployment to GCP:

- `pywin32=303` - Windows specific package, doesn't run on Linux VM's
- `dataclasses==0.8` - GCP env/pip only accepts up to 0.6

# Dashboard Performance

The slowest responding operation is from the data on load coming through BigQuery's API where my Strava data is extracted to using a GCP Cloud Function built under `gcp/strava-data-update-to-bq-cloudfunc`. Other summary results are statically served.

The Kudos prediction vs actual information is coming from real time latest Strava data and sending predicitons to thr Strava Snooper API deployed from the `gcp/kudos-prediction-deployment` folder.

# ML Ops Work Flow

To manage experiments, models and model deployment/version tracking a hosted MLflow instance was built & deployed on GCP. Model deployment is done with dockerized FastAPI endpoint on GCP Cloud Run. THis approach was chosen over Vertex AI as it does not allow scaling to zero instances (as of May 2022) and a single endpoint runs for ~$70/month vs. ~$5/month on Cloud Run.

Work flow details:

1. MLflow Server - deployed on GCP cloud run from `gcp/mlflow-for-gcp`
2. MLflow MySQL database - Cloud SQL mini instance, as this is expensive to run constantly it's turned on/off when developing/refining models
3. MLflow Artifact Store - GCS storage bucket

To begin developing and logging new experiments first startup the Cloud SQL instance which takes ~10-15min to boot up by running:

`make mlflow-launch`

Then from e.g. notebooks, code, etc. use the code below to load the correct environment variables:

```
import mlflow
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.pardir, "gcp", "mlflow-for-gcp", ".env"))

MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME=os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD=os.getenv("MLFLOW_TRACKING_PASSWORD")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"MLFlow Instance to view experiments: {MLFLOW_TRACKING_URI}")
```

Model versioning, updating, tracking etc. is managed all through the UI for putting models into Staging/Production for use by the Strava Snooper API.

**IMPORTANT: Once done working on model development run `make mlflow-shutdown` to shutdown the Cloud SQL instance.**

# Dashboard Deployment: Google App Engine

Production (i.e. public) deployment is setup with CI/CD using Cloud Triggers from publishing versioned new releases from Github with tages of the form `vX.X.X`.

The development server for build testing can be deployed to to validate build and performance. First you must have the Google Cloud SDK installed, if you don't follow the instructions [here](https://cloud.google.com/sdk/docs/).

Once the Cloud SDK is installed run:

`make deploy-dev`

Following deployment you should be able to view the dashboard hosted on GCP by navigating to https://[YOUR GCP PROJECT NAME}.appspot.com or running the following command:

`gcloud app browse`

First you must have the Google Cloud SDK installed, if you don't follow the instructions [here](https://cloud.google.com/sdk/docs/).

Useful details:

- The configuration is managed in `app.yaml` in terms of instance selection, timeouts, etc.
- Make sure you're in the correct GCP project, change using the Cloud SDK with the following command:
- The repo is stored in GCP Cloud Source Repositories to help deal with cloud function deployment issues.

To push changes follow the setup guide form google for authentication here, SDK auth was simplest: [Setting up local authentication](https://cloud.google.com/source-repositories/docs/authentication#windows_1)

Issues with credential storing were an issue which were resolved from [this SOF](https://stackoverflow.com/questions/49473897/git-push-cloud-showing-invalid-authentication-credentials-error) page, the command `git config credential.helper gcloud.cmd` resolved git remote credential issues of incorrect authentication.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make run` or `make deploy`
    ├── README.md          <- The top-level README for developers using this project.
    ├── app.py
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A Sphinx project; see README for how to update/build
    |
    ├── gcp                <- Contains required services deployed to GCP for model tracking with mlflow,
    |                         deploying the StravaSnooper API etc. See specific README's for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-ta-initial-data-exploration`.
    |
    ├── pages              <- Individual pages for the Strava Snooper dashboard, contains componenets from
    |                         src to be able to share common layouts
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip list --format=freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, manipulate or generate data
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │  
    │   ├── pages          <- Scripts to build and generate the individual pages of the dashboard
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---
