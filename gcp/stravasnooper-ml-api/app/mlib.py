import numpy as np
import joblib
import logging
import re
import os
import glob
from typing import List
import mlflow
from dotenv import load_dotenv
import warnings
import xgboost
from google.cloud import storage

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning)


def download_latest_production_models():

    mlflow_env = os.path.join(os.pardir, "mlflow-for-gcp", ".env")
    print(f"[load_latest_production_models]: Loading mlflow .env from {mlflow_env}")
    load_dotenv(mlflow_env)

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # os.environ["MLFLOW_EXPERIMENT_NAME"] = "/kudos-prediction-v1"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    try:
        for mv in client.search_model_versions("name='kudos-prediction'"):
            if dict(mv)["current_stage"] == "Production":
                print(f"[get_latest_production_models]: Found production model.")
                prod_model_details = dict(mv)
    except Exception as e:
        print(f"[get_latest_production_models]: Error - {e}")

    storage_client = storage.Client()
    bucket = storage_client.bucket("stravasnooper-mlflow-artifacts")

    current_stage = prod_model_details["current_stage"]
    gcs_bucket = re.search("gs://(.+?)/", prod_model_details["source"]).group(1)
    model_path = prod_model_details["source"].lstrip(f"gs://{gcs_bucket}/")

    # file path
    blob_path = (
        model_path + "/model.xgb"
    )  # prod_model_details["source"].lstrip(f"gs://{model_path}/")
    blob = bucket.blob(blob_path)

    # find previous version of xgb model
    model_output_path = os.path.join(os.getcwd(), "models", "kudos-prediction")
    for file in os.listdir(model_output_path):
        if file.endswith(".xgb"):
            os.remove(os.path.join(model_output_path, file))

    blob.download_to_filename(
        os.path.join(
            model_output_path,
            f"model_ver{prod_model_details['version']}.xgb",
        )
    )


def load_kudos_model(
    model_path=os.path.join(os.getcwd(), "models", "kudos-prediction")
):
    latest_model_ver = -1
    latest_model = None
    for mdl in glob.glob(os.path.join(model_path, "model_ver*.xgb")):
        model_version = int(re.search("model_ver(\d+).xgb", mdl).group(1))
        if model_version > latest_model_ver:
            latest_model_ver = model_version
            latest_model = mdl
    print(
        f"[load_kudos_model]: Loading kudos-prediction model version {latest_model_ver}"
    )
    kudos_model = xgboost.XGBRegressor()
    kudos_model.load_model(latest_model)

    return kudos_model


def format_input(
    custom_name_bool: List[int],
    distance_km: List[int],
    achievement_count: List[int],
    total_elevation_gain: List[int],
):

    vals = np.array(
        list(
            zip(custom_name_bool, distance_km, achievement_count, total_elevation_gain)
        )
    )

    return vals


def predict(
    custom_name_bool: List[int],
    distance_km: List[int],
    achievement_count: List[int],
    total_elevation_gain: List[int],
    num_followers: int,
):

    mdl = load_kudos_model()
    values = format_input(
        custom_name_bool=custom_name_bool,
        distance_km=distance_km,
        achievement_count=achievement_count,
        total_elevation_gain=total_elevation_gain,
    )

    perc_followers = mdl.predict(values).round(3)
    num_kudos = (num_followers * perc_followers).round().astype(int)

    return perc_followers.tolist(), num_kudos.tolist()
