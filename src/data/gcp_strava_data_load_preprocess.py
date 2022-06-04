import os
import glob
from datetime import datetime, timedelta
import json
import time
import logging

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgbm
from google.cloud import bigquery


def load_strava_activity_data_from_bq(users=["TyAndrews"]):
    """Gets the Strava data from Bigquery for single or multiple users in dataframe format.

    Args:
        users (list, optional): The users for which to load Strava data from Bigquery. Defaults to ["TyAndrews"].

    Returns:
        DataFrame: All users data in a single dataframe with the column "name" to denote between users.
    """

    start_time = time.time()

    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    # for local development if GCP server not available
    if GCP_PROJECT_ID == None:
        GCP_PROJECT_ID = "stravasnooper-dev"

    processed_df = pd.DataFrame()
    for user in users:

        bqclient = bigquery.Client(project=GCP_PROJECT_ID)

        strava_data_query = """
            SELECT 
                name,
                distance_km,
                type,
                start_date_local AS start_time,
                distance_km AS distance_raw_km,
                elapsed_time_hrs AS elapsed_time_raw_hrs,
                moving_time_hrs AS moving_time_raw_hrs,
                total_elevation_gain AS elevation_gain, 
                kudos_count AS kudos,
                achievement_count,

            FROM `{0}.prod_dashboard.raw_strava_data` 
            ORDER BY start_date_local DESC
            LIMIT 5000""".format(
            bqclient.project
        )

        try:
            raw_df = (
                bqclient.query(strava_data_query, timeout=15)
                .result(job_retry=None)
                .to_dataframe()
            )

        except Exception as e:
            logging.error(
                f"[load_strava_activity_data_from_bq] Issue getting data from BQ, using static data to serve dashboard - {e}"
            )
            static_data = os.path.join(
                os.getcwd(),
                "data",
                "processed",
                "TyAndrews_ProcessedStravaData_19-02-2021.csv",
            )
            raw_df = pd.read_csv(static_data)
        finally:
            proc_user_df = preprocess_strava_df(raw_df)
            proc_user_df["user"] = user
            processed_df = pd.concat(
                [processed_df, proc_user_df], axis=0, ignore_index=True
            )

            logging.info(
                f"[load_strava_activity_data]: Took {time.time() - start_time: .2f}s to get BQ data"
            )

            return processed_df


def preprocess_strava_df(raw_df, min_act_length=1200, max_act_dist=400, export=False):
    """Processes the raw BigQuery data and manages feature additions, scaling etc. for use in the dashboard.

    Args:
        raw_df (DataFrame): The extracted data from BigQuery
        min_act_length (int, optional): THe shortest activity to include in the results in seconds. Defaults to 1200 (20min).
        max_act_dist (int, optional): Longest activity allowed to catch data issues/outliers, in km . Defaults to 400.
        export (bool, optional): Used to export the processed df to CSV for local work/offline prototyping. Defaults to False.

    Returns:
        DataFrame: The cleaned and processed data with additional features etc. added.
    """

    processed_df = raw_df.copy(deep=True)

    processed_df["custom_name_bool"] = 1
    processed_df.loc[
        processed_df.name.isin(
            ["Afternoon Ride", "Lunch Ride", "Morning Ride", "Evening Ride"]
        ),
        "custom_name_bool",
    ] = 0

    if export == True:
        processed_df.to_csv(r"data\processed\ProcessedStravaData.csv")

    return processed_df


def load_employment_model_data():
    """Loads the pre-generated employment model data for use in the dashboard.

    Returns:
        DataFrame: Tuple, training employment data first, then test employment data
    """

    model_data_file_path = os.path.abspath(
        os.path.join(os.getcwd(), "data", "processed")
    )
    logging.info("Loading Employment Model Data: " + model_data_file_path)
    start = time.process_time()

    train_data = os.path.join(model_data_file_path, "train_employment_data.csv")
    test_data = os.path.join(model_data_file_path, "test_employment_data.csv")

    files = [train_data, test_data]
    all_data = []
    for f in files:
        data = pd.read_csv(f)

        all_data.append(data)

    return all_data[0], all_data[1]


def preprocess_employment_model_data(input_data, work_hours):
    """Breaks out hourly percentage of activities started and uses the currently set work hours
    to return breakdown of morning vs. afternoon activities.

    Args:
        input_data (DataFrame): Raw data with 0-23 columns for each hour of the day, label, year and month columns.
        work_hours (List): A list of two windows for morning/aft work hours, e.g. [[9, 11], [13, 16]]

    Returns:
        DataFrame, Series: Tuple of the processed dataframe with new columns, and the labels corresponding to the data.
    """

    data = input_data.iloc[:, 0:24]
    labels = input_data["label"]

    morning = work_hours[0]
    afternoon = work_hours[1]

    data["morn"] = data.iloc[:, morning[0] - 1 : morning[1]].sum(axis=1)
    data["aft"] = data.iloc[:, afternoon[0] - 1 : afternoon[1]].sum(axis=1)

    return data, labels


def load_week_start_times_data():
    """To display differences in start times between full years of employed vs. not employed
    the static data is loaded.

    Returns:
        DataFrame: Yearly data with activities by day of week 0-6 and hours of the day 0-23.
    """

    logging.info("Loading Weekly Employment Summary Data: ")

    week_data = os.path.join(
        os.getcwd(), "data", "processed", "yearly_week_start_times.json"
    )

    yearly_week_summary_data = json.load(open(week_data, "r"))

    return yearly_week_summary_data


def load_lgbm_model(model_file_name="lgbm_employment_classifier.txt"):
    """To use the lgbm employment classifier model first load with lgbm package.

    Args:
        model_file_name (str, optional): Model name in models. Defaults to "lgbm_employment_classifier.txt".

    Returns:
        lgbm.Booster: LGBM Booster model
    """

    lgbm_model = lgbm.Booster(
        model_file=os.path.join(os.getcwd(), "models", model_file_name)
    )

    return lgbm_model


def load_logreg_model(model_file_name="logreg_employment_model.pkl"):
    """Load to SK learn object the pickled employment prediction logistic regression model.

    Args:
        model_file_name (str, optional): Name of model file in models folder. Defaults to "logreg_employment_model.pkl".

    Returns:
        LogisticRegression: SKlearn logistic regression model instance.
    """

    logreg_model = pickle.load(
        open(os.path.join(os.getcwd(), "models", model_file_name), "rb")
    )

    return logreg_model


def load_logreg_model_results(data_set):
    """Load static model employment predictions for faster dashboard loading performance.

    Args:
        data_set (str): "test" or "train" dataset to load.

    Returns:
        DataFrame: Employment logistic regression model predictions - index, prediction, label e.g. 12,1,0
    """

    logging.info("Loading " + data_set + " LogReg Model Results: ")

    if data_set == "test":
        logreg_results = pd.read_csv(
            glob.glob(
                os.path.join(os.getcwd(), "data", "processed", "test_logreg_model*.csv")
            )[0]
        )
    elif data_set == "train":
        logreg_results = pd.read_csv(
            glob.glob(
                os.path.join(
                    os.getcwd(), "data", "processed", "train_logreg_model*.csv"
                )
            )[0]
        )

    return logreg_results


def load_lgbm_model_results(data_set):
    """Load static lgbm model employment predictions for faster dashboard loading performance.

    Args:
        data_set (str): "test" or "train" dataset to load.

    Returns:
        DataFrame: Employment LGBM model predictions - index, prediction, label e.g. 12,1,0
    """
    print("Loading " + data_set + " LogReg Model Results: ")

    if data_set == "test":
        lgbm_results = pd.read_csv(
            glob.glob(
                os.path.join(os.getcwd(), "data", "processed", "test_lgbm_model*.csv")
            )[0]
        )
    elif data_set == "train":
        lgbm_results = pd.read_csv(
            glob.glob(
                os.path.join(os.getcwd(), "data", "processed", "train_lgbm_model*.csv")
            )[0]
        )

    return lgbm_results


def load_lgbm_heatmap(data_set):
    """Load heatmap of confidence values of empl. vs. un-empl. time based on perc.
    of activities started during morning/afternoon work hours.

    Args:
        data_set (str ): "test" or "train" dataset heat maps.

    Returns:
        DataFrame: LGBM heat map values of form - index, X, Y, Z, where Z is prediction confidence
                for heat map, X/Y are morning/aft work hours activities started
    """

    lgbm_heatmap = pd.read_csv(
        glob.glob(
            os.path.join(
                os.getcwd(), "data", "processed", data_set + "_lgbm_model_heatmap*.csv"
            )
        )[0]
    )

    return lgbm_heatmap


def load_logreg_heatmap(data_set):
    """Load heatmap of confidence values of empl. vs. un-empl. time based on perc.
    of activities started during morning/afternoon work hours.

    Args:
        data_set (str): "test" or "train" dataset heat maps.

    Returns:
        DataFrame: Logisitic Regression heat map values of form - index, X, Y, Z, where Z is prediction confidence
                for heat map, X/Y are percentage of activities started during morning/aft work hours activities
    """

    logreg_heatmap = pd.read_csv(
        glob.glob(
            os.path.join(
                os.getcwd(),
                "data",
                "processed",
                data_set + "_logreg_model_heatmap*.csv",
            )
        )[0]
    )

    return logreg_heatmap


def generate_employment_prediction_model_data(
    activity_df, start_year, end_year, start_month, end_month, label
):
    """Used to generate month by month data of activities per hour of the day, both in terms of percentage
    of activites started at each hour/day of the week, as well as raw counts of activities per hour  of the
    day for training the employment prediction models.

    Args:
        activity_df (DataFrame): The pre-processed Strava activity data frame.
        start_year (int): Year in which to start processing data.
        end_year (int): YEar in which to stop processing data from Strava data.
        start_month (int): Month for which to start processing Strava data.
        end_month (int): Month for which to end processing of strava data in end_year
        label (int): 0 for unemployed, 1 for employed

    Returns:
        summary_data (dict): indexed by year, month, day of week, hour and how many activities were started then
        train_data (list): activities by hour of day 0-23, then label, year, month.
    """

    summary_data = {}

    train_data = []

    for year in range(start_year, end_year + 1):

        summary_data[str(year)] = {}

        begin_month = 1
        stop_month = 12 + 1  # Add one to account for indexing up to but not including

        if year == start_year:
            begin_month = start_month
        if year == end_year:
            stop_month = (
                end_month + 1
            )  # Add one to account for indexing up to but not including

        print(f"{year} {begin_month} {stop_month}")

        for month in range(begin_month, stop_month):

            summary_data[str(year)][str(month)] = {}

            print(f"\t{month}")

            # for month in range(quarter*3, quarter*3+3):

            #     print(f'\t\t{month}')

            for day in range(0, 7):
                # print(f'\tProcessing Day {day}')

                summary_data[str(year)][str(month)][str(day)] = 24 * [0]

                # days_data = quarter_data[pd.DatetimeIndex(quarter_data.start_date_local).weekday == day]

                for hour in range(0, 24):

                    # print(f'\t\tAccumulating Hour {hour}')

                    # hours_data = days_data.set_index('start_date_local')[pd.DatetimeIndex(days_data.start_date_local).hour == hour]
                    hours_data = activity_df[
                        (pd.DatetimeIndex(activity_df.start_date_local).year == year)
                        & (
                            pd.DatetimeIndex(activity_df.start_date_local).month
                            == month
                        )
                        & (
                            pd.DatetimeIndex(activity_df.start_date_local).weekday
                            == day
                        )
                        & (pd.DatetimeIndex(activity_df.start_date_local).hour == hour)
                    ]

                    summary_data[str(year)][str(month)][str(day)][hour] = len(
                        hours_data
                    )

            week_days = np.array(24 * [0])

            # Calculate what eprcentag of workout start times occur at each hour in the day.
            for day in range(0, 5):
                week_days += summary_data[str(year)][str(month)][str(day)]
            week_days_perc = week_days / sum(week_days)

            month_data = np.append(week_days_perc, [label, year, month])

            train_data.append(month_data)
            # week_days_perc = pd.DataFrame(data=week_days_perc, columns=['exercise_start'])

    return summary_data, train_data


def generate_weekly_start_time_dict(activity_df, year):
    """Builds data object of how many activities were started per hour of the day in a given year.

    Args:
        activity_df (DataFrame): Processed strava dataframe
        year (int): the year for which to generate the time of day activity summaries

    Returns:
        dict: Indexed by day of week (0-6) and hour of day (0-23) gives count of activities during that time.
    """

    week_summary_data = {}

    # week_summary_data[str(year)] = {}

    years_data = activity_df[
        pd.DatetimeIndex(activity_df.start_date_local).year == year
    ]

    for day in range(0, 7):
        # print(f'\tProcessing Day {day}')

        week_summary_data[str(day)] = 24 * [0]

        days_data = years_data[
            pd.DatetimeIndex(years_data.start_date_local).weekday == day
        ]

        for hour in range(0, 24):

            # print(f'\t\tAccumulating Hour {hour}')

            hours_data = days_data.set_index("start_date_local_raw")[
                pd.DatetimeIndex(days_data.start_date_local).hour == hour
            ]

            week_summary_data[str(day)][hour] = len(hours_data)

    return week_summary_data
