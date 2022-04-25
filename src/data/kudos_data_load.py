import os
import glob
from datetime import datetime, timedelta
import json
import time

import pandas as pd
import numpy as np


def load_static_kudos_predictions(date_string):

    file_path = os.path.join(
        # "..",
        "data",
        "processed",
        "kudos_prediction",
        f"kudos_model_raw_data_{date_string}.csv",
    )

    return pd.read_csv(file_path)
