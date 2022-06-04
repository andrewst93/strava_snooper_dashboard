import os
import glob
from datetime import datetime, timedelta
import json
import time

import pandas as pd
import numpy as np


def load_static_kudos_predictions(date_string):
    """Imports the pre-generated kudos predictions used for the interactive visualizations/selectors
    of the kudos prediction dashboard.

    Args:
        date_string (str): The data of the most up to data static data file in data/kudos-prediction

    Returns:
        DataFrame: the imported DataFrame containing the predictions byt features at pre-generated steps.
    """

    file_path = os.path.join(
        # "..",
        "data",
        "processed",
        "kudos_prediction",
        f"kudos_model_raw_data_{date_string}.csv",
    )

    return pd.read_csv(file_path)
