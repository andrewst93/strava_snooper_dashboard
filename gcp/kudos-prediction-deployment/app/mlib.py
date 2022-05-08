import numpy as np
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(model=os.path.join(os.getcwd(), "models", "model.joblib")):

    mdl = joblib.load(model)

    return mdl


def format_input(
    custom_name_bool: int,
    distance_km: int,
    achievement_count: int,
    total_elevation_gain: int,
):

    val = np.array(
        [[custom_name_bool, distance_km, achievement_count, total_elevation_gain]]
    )

    return val


def predict(
    custom_name_bool: int,
    distance_km: int,
    achievement_count: int,
    total_elevation_gain: int,
    num_followers: int,
):

    mdl = load_model()
    values = format_input(
        custom_name_bool=custom_name_bool,
        distance_km=distance_km,
        achievement_count=achievement_count,
        total_elevation_gain=total_elevation_gain,
    )

    perc_followers = mdl.predict(values).round(3)
    num_kudos = num_followers * perc_followers

    predict_result = {
        "num_followers": num_followers,
        "perc_followers": perc_followers.round(3).tolist(),
        "num_kudos": num_kudos.round().astype(int).tolist(),
    }
    logging.debug(f"Prediction: {predict_result}")

    return predict_result
