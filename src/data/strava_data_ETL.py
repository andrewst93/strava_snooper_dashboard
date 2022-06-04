## OUT OF DATE - latest in strava-data-update-to-bq-cloudfunc

# import os
# import csv
# import json
# import time
# import requests
# import datetime as dt
# import pandas as pd
# import datetime as dt

# from google.cloud import bigquery
# from dotenv import load_dotenv
# load_dotenv()

# STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
# STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
# STRAVA_TOKEN = json.loads(os.getenv('STRAVA_TOKEN').replace("\'", "\""))

# def get_latest_request_date():

#     with open(os.path.join('data', 'raw', 'request_log.csv'), 'r') as f:
#         # read file line-by-line
#         lines = f.read().splitlines()
#         # store last line as a dictionary
#         first_line = lines[0].split(',')
#         last_line = lines[-1].split(',')
#         last_line_dict = dict(list(zip(first_line, last_line)))
#         # extract timestamp from last line
#         start_date = last_line_dict['timestamp']
#     # convert timestamp from ISO-8601 to UNIX format
#     start_date_dt = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
#     start_date_tuple = start_date_dt.timetuple()
#     start_date_unix = int(time.mktime(start_date_tuple))

#     return start_date_unix

# def get_new_strava_activity_data():

#     global STRAVA_TOKEN
#     # Check current token is still valid and update if needed
#     if STRAVA_TOKEN['expires_at'] < time.time():
#         # Make Strava auth API call with current refresh token
#         response = requests.post(
#                             url = 'https://www.strava.com/oauth/token',
#                             data = {
#                                     'client_id': STRAVA_CLIENT_ID,
#                                     'client_secret': STRAVA_CLIENT_SECRET,
#                                     'grant_type': 'refresh_token',
#                                     'refresh_token': STRAVA_TOKEN['refresh_token']
#                                     }
#                         )
#         # Save response as json in new variable
#         new_strava_token = response.json()
#         # Use new Strava tokens from now
#         STRAVA_TOKEN = new_strava_token

#     # store URL for activities endpoint
#     base_url = "https://www.strava.com/api/v3/"
#     endpoint = "athlete/activities"
#     url = base_url + endpoint
#     start_date_unix = get_latest_request_date()
#     # define headers and parameters for request
#     headers = {"Authorization": "Bearer {}".format(STRAVA_TOKEN['access_token'])}
#     params = {"after": start_date_unix,
#             "per_page": 200,
#             "page" : str(1)
#             }
#     # make GET request to Strava API
#     req = requests.get(url, headers = headers, params = params).json()

#     return pd.json_normalize(req)

# def clean_raw_strava_data(raw_df, min_act_length=1200, max_act_dist=400, export=False):

#     if len(raw_df) == 0:
#         print("Raw data frame has no entries to clean, check qery and df passed in.")
#         return None
#     print(raw_df.head())

#     cleaned_df = raw_df.copy(deep=True)
#     cleaned_df = cleaned_df.drop(cleaned_df[(cleaned_df.elapsed_time > min_act_length) & (cleaned_df.distance < max_act_dist)].index)

#     print(f'\t{len(raw_df[(raw_df.elapsed_time < min_act_length) & (raw_df.distance < max_act_dist)])} Activities Under 20min in Length, Removed from Dataset')
#     cleaned_df = cleaned_df.drop(columns=['athlete.resource_state', 'display_hide_heartrate_option', 'map.resource_state', 'map.summary_polyline', 'heartrate_opt_out',
#                                 'display_hide_heartrate_option', 'resource_state'])
#     # cleaned_df = cleaned_df.convert_dtypes()
#     cleaned_df[['distance_km']] = cleaned_df[['distance']]/1000
#     cleaned_df[['start_date_local']] = pd.to_datetime(cleaned_df['start_date_local'])
#     cleaned_df[['start_date']] = pd.to_datetime(cleaned_df['start_date'])
#     cleaned_df['exer_start_time'] = pd.to_datetime(pd.to_datetime(cleaned_df['start_date_local']).dt.strftime('1990:01:01:%H:%M:%S'), format='1990:01:01:%H:%M:%S')
#     cleaned_df['act_type_perc_time'] = (cleaned_df['moving_time']/sum(cleaned_df['moving_time']))
#     cleaned_df['elapsed_time_hrs'] = (cleaned_df['elapsed_time']/3600)
#     cleaned_df['moving_time_hrs'] = (cleaned_df['moving_time']/3600)
#     # cleaned_df['distance_km'] = (cleaned_df['distance']/1000)

#     cleaned_df.columns = cleaned_df.columns.str.replace("[.]", "_")


#     return cleaned_df

# def load_strava_data_into_bq(df, table_id):

#     client = bigquery.Client(project='stavasnooper')
#     # filename = '/path/to/file/in/nd-format.json'
#     dataset_id = 'test'

#     dataset_ref = client.dataset(dataset_id)
#     table_ref = dataset_ref.table(table_id)
#     job_config = bigquery.LoadJobConfig()
#     # job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
#     job_config.autodetect = True

#     job = client.load_table_from_dataframe(
#             df,
#             table_ref,
#             location="us", # Must match the destination dataset location.
#             job_config=job_config,
#     )  # API request
#     i=0
#     while client.get_job(job.job_id, location='us').running():
#         print("Running Load Job", "."*i, '\r',)
#         i+=1
#         time.sleep(1)

#     print(f"Done Load Job.")

# def update_request_log(n_activities):

#     # storing current date
#     current_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     # updating request log file
#     with open(os.path.join("..", 'data', 'raw', 'request_log.csv'), 'a', newline = '') as f:
#         csv_writer = csv.writer(f)
#         csv_writer.writerow([current_date, n_activities])

# def main():

#     raw_df = get_new_strava_activity_data()

#     if len(raw_df) > 0:

#         cleaned_df = clean_raw_strava_data(raw_df)

#         load_strava_data_into_bq(cleaned_df, 'strava_cleaned')

#         update_request_log(n_activities=len(cleaned_df))
#     else:
#         print('No activities since last run or an error has occured.')


# if __name__ == "__main__":

#     main()
