# Strava Snooper Dashboard

Digging deeper into personal Strava data to figure out how training has changed over the years and seeing what insights Strava might be able to gleen from all our data.

# Requirements

The primary dashboard and model development/training is encompassed in the requirements.txt for all development and tools for deployment.

To ensure the requirements.txt is updated in the format expected by App Engine use the following command to generate one in the correct format from a conda environment:

`pip list --format=freeze > requirements.txt`

On windows PyWin32 is used but is not included in the requirements due to it breaking the App Engine deployment. The version used is `pywin32==303`.

# Dashboard Performance

Majority of the visualizations are fed by preprocessed data and prediction/evaluation metrics. This was initially setup to run real time but with on demand loading of the page on the free tier of App Engine meant it would load much too slow.

Future improvements are connecting with BigQuery and having live up to date data fed into the dashboard which is partially done.

# Google App Engine Deployment

First you must have the Google Cloud SDK installed, if you don't follow the instructions [here](https://cloud.google.com/sdk/docs/).

The configuration is managed in `app.yaml` in terms of instance selection, timeouts, etc.

Make sure you're in the correct GCP project, change using the Cloud SDK with the following command:

`gcloud config set project [YOUR GCP PROJECT NAME]`

To deploy run the following command from within the repo:

`gcloud app deploy`

Following deployment you should be able to view the dashboard hosted on GCP by navigating to https://[YOUR GCP PROJECT NAME}.appspot.com or running the following command:

`gcloud app browse`
