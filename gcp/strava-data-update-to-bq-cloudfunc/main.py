import base64
import os
import csv
import json
import time
import requests
import datetime as dt
import pandas as pd
import datetime as dt
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_TOKEN = json.loads(os.getenv("STRAVA_TOKEN").replace("'", '"'))
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")


def get_latest_request_date():
    """Queries bigquery for the last prcessed timestamp.

    Returns:
        [unix timestamp]: The number of milliseconds that have passed since January 1, 1970 00:00:00 (UTC)
    """

    client = bigquery.Client(project="stavasnooper")
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)

    max_request_date_query = """SELECT MAX(processed_timestamp) AS most_recent_processing
    FROM `stavasnooper.{0}.{1}` LIMIT 1000""".format(
        DATASET_ID, TABLE_ID
    )

    try:
        latest_processing_date = (
            client.query(max_request_date_query).result().to_dataframe()
        )
        latest_processing_date = latest_processing_date.most_recent_processing.iloc[0]
        print(latest_processing_date)
        if pd.isnull(latest_processing_date):
            print(
                "get_latest_request_date: No processing timestamp in biquery table, using 2000-01-01 00:00:00"
            )
            latest_processing_date = dt.datetime(2000, 1, 1)
        print(
            "get_latest_request_date: Last processed timestamp - {0}".format(
                latest_processing_date
            )
        )

        start_date_tuple = latest_processing_date.timetuple()
        start_date_unix = int(time.mktime(start_date_tuple))

    except Exception as e:
        print(f"Issue with getting event log query results:\n {e}")

    return start_date_unix


def clean_raw_strava_data(raw_df, min_act_length=300, max_act_dist=400):
    """Preprocessing for the raw strava data returned by the Strava V3 API

    Args:
        raw_df (DataFrame): The raw data as returned by the Strava V3 API
        min_act_length (int, optional): Shortest activity allowed in seconds. Defaults to 300 (5min).
        max_act_dist (int, optional): Longest activity allowed in km and is used to remove outliers. Defaults to 400km.

    Returns:
        DataFrame: Completely processed Strava data
    """

    if len(raw_df) == 0:
        print(
            "clean_raw_strava_data: Raw data frame has no entries to clean, check query and df passed in."
        )
        return None

    cleaned_df = raw_df.copy(deep=True)
    cleaned_df = cleaned_df.drop(
        cleaned_df[
            (cleaned_df.elapsed_time < min_act_length)
            | (cleaned_df.distance > max_act_dist * 1000)
        ].index
    )

    print(
        "clean_raw_strava_data: {} Activities Under {} min in Length or greater than {}km distance, Removed from Dataset".format(
            len(
                raw_df[
                    (raw_df.elapsed_time < min_act_length)
                    | (raw_df.distance > max_act_dist * 1000)
                ]
            ),
            min_act_length / 60,
            max_act_dist,
        )
    )

    # remove unused settings/fields returned by the API
    cleaned_df = cleaned_df.drop(
        columns=[
            "athlete.resource_state",
            "display_hide_heartrate_option",
            "map.resource_state",
            "map.summary_polyline",
            "heartrate_opt_out",
            "display_hide_heartrate_option",
            "resource_state",
        ]
    )

    # correct units/timestamps for key fields
    cleaned_df[["distance_km"]] = cleaned_df[["distance"]] / 1000
    cleaned_df[["start_date_local"]] = pd.to_datetime(cleaned_df["start_date_local"])
    cleaned_df[["start_date"]] = pd.to_datetime(cleaned_df["start_date"])
    cleaned_df["exer_start_time"] = pd.to_datetime(
        pd.to_datetime(cleaned_df["start_date_local"]).dt.strftime(
            "1990:01:01:%H:%M:%S"
        ),
        format="1990:01:01:%H:%M:%S",
    )
    cleaned_df["act_type_perc_time"] = cleaned_df["moving_time"] / sum(
        cleaned_df["moving_time"]
    )
    cleaned_df["elapsed_time_hrs"] = cleaned_df["elapsed_time"] / 3600
    cleaned_df["moving_time_hrs"] = cleaned_df["moving_time"] / 3600

    # append when this processing was run
    cleaned_df[
        "processed_timestamp"
    ] = dt.datetime.now()  # .strftime('%Y-%m-%d %H:%M:%S')

    cleaned_df.columns = cleaned_df.columns.str.replace("[.]", "_")

    return cleaned_df


def load_strava_data_into_bq(df):
    """Loads the cleaned/preprocessed Strava data into BigQuery

    Args:
        df (DataFrame): Cleaned & pre-processed Strava data returned from clean_raw_strava_data
    """

    client = bigquery.Client(project="stavasnooper")

    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_ID)
    job_config = bigquery.LoadJobConfig()
    # job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.autodetect = True

    job = client.load_table_from_dataframe(
        df,
        table_ref,
        location="us",  # Must match the destination dataset location.
        job_config=job_config,
    )  # API request

    while client.get_job(job.job_id, location="us").running():
        time.sleep(1)

    print("load_strava_data_into_bq: Done Load Job for {0} activities.".format(len(df)))


def get_new_strava_activity_data():
    """Primary function to coordinate, requesting new Strava data, cleaning it and uploading to BigQuery"""

    global STRAVA_TOKEN
    # Check current token is still valid and update if needed
    if STRAVA_TOKEN["expires_at"] < time.time():
        # Make Strava auth API call with current refresh token
        response = requests.post(
            url="https://www.strava.com/oauth/token",
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": STRAVA_TOKEN["refresh_token"],
            },
        )
        # Save response as json in new variable
        new_strava_token = response.json()
        # Use new Strava tokens from now
        STRAVA_TOKEN = new_strava_token
        print("get_new_activity_data: strava token updated")

    # store URL for activities endpoint
    base_url = "https://www.strava.com/api/v3/"
    endpoint = "athlete/activities"
    url = base_url + endpoint

    start_date_unix = get_latest_request_date()

    # define headers and parameters for request
    headers = {"Authorization": "Bearer {}".format(STRAVA_TOKEN["access_token"])}

    num_activities = 0
    num_clean_activities = 0
    page = 1

    # get pages of the Strava data until no more acitvities available after the start_date_unix that was last processed
    while True:
        params = {"after": start_date_unix, "per_page": 200, "page": str(page)}
        # make GET request to Strava API
        req = requests.get(url, headers=headers, params=params).json()

        # if no results then exit loop
        if not req:
            break

        df = pd.json_normalize(req)
        print(
            "get_new_activity_data: strava data received, {0} activities".format(
                len(df)
            )
        )
        # df_res = df_res.append(df)

        cleaned_df = clean_raw_strava_data(df)
        print(
            "get_new_activity_data: activity data cleaned, {0} activities remaining.".format(
                len(cleaned_df)
            )
        )

        load_strava_data_into_bq(cleaned_df)
        print(
            "get_new_activity_data: activity data loaded to BQ, page {0} had {1} activities processed.".format(
                page, len(df)
            )
        )

        # move to next page
        page += 1
        num_activities += len(df)
        num_clean_activities += len(cleaned_df)

    print(
        "get_new_activity_data: Done processing all {0} activities, {1} clean,  since last refresh.".format(
            num_activities, num_clean_activities
        )
    )


def main(event, context):

    get_new_strava_activity_data()
