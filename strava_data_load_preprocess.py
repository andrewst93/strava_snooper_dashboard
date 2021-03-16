import os
import glob
from datetime import datetime, timedelta
import json
import time

import pandas as pd 
import numpy as np
import pickle
import lightgbm as lgbm


def load_strava_activity_data(data_file_path):
    raw_files_dict = {}
    for f in glob.glob(os.path.join(data_file_path, "*ProcessedStravaData*.csv")):
        user = os.path.basename(f).split('_')[0]
        print(f'{user} Data Found')
        # raw_files_dict[user] = preprocess_strava_df(pd.read_csv(f))
        raw_files_dict[user] = pd.read_csv(f)

    return raw_files_dict

def preprocess_strava_df(raw_df, min_act_length=1200, max_act_dist=400, export=False):

    # Remove activites under 5 minutes in length
    processed_df = raw_df[(raw_df.elapsed_time_raw > min_act_length) & (raw_df.distance < max_act_dist)]
    print(f'\t{len(raw_df[(raw_df.elapsed_time_raw < min_act_length) & (raw_df.distance < max_act_dist)])} Activities Under 20min in Length, Removed from Dataset')

    processed_df = processed_df.convert_dtypes()
    processed_df[['distance','distance_raw']] = processed_df[['distance','distance_raw']].apply(pd.to_numeric)
    processed_df[['start_date_local']] = pd.to_datetime(processed_df['start_date_local_raw'], unit='s') #.apply(pd.to_datetime(unit='s'))
    processed_df['exer_start_time'] = pd.to_datetime(processed_df['start_date_local'].dt.strftime('1990:01:01:%H:%M:%S'), format='1990:01:01:%H:%M:%S')
    # processed_df['exer_start_time'] = pd.to_datetime(pd.to_datetime(processed_df['start_time']).dt.strftime('1990:01:01:%H:%M:%S'), format='1990:01:01:%H:%M:%S')
    processed_df['exer_start_time'] = processed_df['exer_start_time'].dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    processed_df['act_type_perc_time'] = (processed_df['moving_time_raw']/sum(processed_df['moving_time_raw']))
    processed_df['elapsed_time_raw_hrs'] = (processed_df['elapsed_time_raw']/3600)
    processed_df['moving_time_raw_hrs'] = (processed_df['moving_time_raw']/3600)
    processed_df['distance_raw_km'] = (processed_df['distance_raw']/1000)

    if export == True:
        processed_df.to_csv(r"data\ProcessedStravaData.csv")

    return processed_df

def load_employment_model_data():

    model_data_file_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    print('Loading Employment Model Data: ' + model_data_file_path)
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

    data = input_data.iloc[:,0:24]
    labels = input_data['label']

    morning = work_hours[0]
    afternoon = work_hours[1]

    data['morn'] = data.iloc[:, morning[0]-1:morning[1]].sum(axis=1)
    data['aft'] = data.iloc[:, afternoon[0]-1:afternoon[1]].sum(axis=1)
    
    return data, labels

def load_week_start_times_data():

    print('Loading Weekly Employment Summary Data: ')

    week_data = os.path.join(os.getcwd(), 'data', "yearly_week_start_times.json")

    yearly_week_summary_data = json.load(open(week_data, "r"))

    return yearly_week_summary_data

def load_lgbm_model(model_file_name="lgbm_employment_classifier.txt"):

    lgbm_model = lgbm.Booster(model_file=os.path.join(os.getcwd(), 'models', model_file_name))

    return lgbm_model

def load_logreg_model(model_file_name="logreg_employment_model.pkl"):

    logreg_model  = pickle.load(open(os.path.join(os.getcwd(), 'models', model_file_name), 'rb'))

    return logreg_model

def load_logreg_model_results(data_set):

    print('Loading ' + data_set + ' LogReg Model Results: ')

    if data_set == 'test':
        logreg_results = pd.read_csv( glob.glob(os.path.join(os.getcwd(),'data', "test_logreg_model*.csv"))[0])
    elif data_set == 'train':
        logreg_results = pd.read_csv( glob.glob(os.path.join(os.getcwd(),'data', "train_logreg_model*.csv"))[0])

    return logreg_results

def load_lgbm_model_results(data_set):

    print('Loading ' + data_set + ' LogReg Model Results: ')

    if data_set == 'test':
        lgbm_results = pd.read_csv( glob.glob(os.path.join(os.getcwd(),'data', "test_lgbm_model*.csv"))[0])
    elif data_set == 'train':
        lgbm_results = pd.read_csv( glob.glob(os.path.join(os.getcwd(),'data', "train_lgbm_model*.csv"))[0])

    return lgbm_results

def load_lgbm_heatmap(data_set):

    lgbm_heatmap = pd.read_csv(glob.glob(os.path.join(os.getcwd(),'data', data_set + "_lgbm_model_heatmap*.csv"))[0])

    return lgbm_heatmap

def load_logreg_heatmap(data_set):

    logreg_heatmap = pd.read_csv(glob.glob(os.path.join(os.getcwd(),'data', data_set + "_logreg_model_heatmap*.csv"))[0])

    return logreg_heatmap


def generate_employment_prediction_model_data(activity_df, start_year, end_year, start_month, end_month, label):

    summary_data = {}

    train_data = []

    for year in range(start_year, end_year+1):

        summary_data[str(year)] = {}

        begin_month = 1
        stop_month = 12 + 1 #Add one to account for indexing up to but not including

        if year == start_year: begin_month = start_month
        if year == end_year: stop_month = end_month + 1 #Add one to account for indexing up to but not including

        print(f'{year} {begin_month} {stop_month}')

        for month in range(begin_month, stop_month):

            summary_data[str(year)][str(month)] = {}

            print(f'\t{month}')

            # for month in range(quarter*3, quarter*3+3):
                    
            #     print(f'\t\t{month}')

            for day in range(0,7):
                # print(f'\tProcessing Day {day}')

                summary_data[str(year)][str(month)][str(day)] = 24*[0]

                # days_data = quarter_data[pd.DatetimeIndex(quarter_data.start_date_local).weekday == day]
                
                for hour in range(0,24):

                    # print(f'\t\tAccumulating Hour {hour}')

                    # hours_data = days_data.set_index('start_date_local')[pd.DatetimeIndex(days_data.start_date_local).hour == hour]
                    hours_data = activity_df[ (pd.DatetimeIndex(activity_df.start_date_local).year == year) & 
                                    (pd.DatetimeIndex(activity_df.start_date_local).month == month) &
                                    (pd.DatetimeIndex(activity_df.start_date_local).weekday == day) &
                                    (pd.DatetimeIndex(activity_df.start_date_local).hour == hour)]

                    summary_data[str(year)][str(month)][str(day)][hour] = len(hours_data)

            week_days = np.array(24*[0])

            # Calculate what eprcentag of workout start times occur at each hour in the day.
            for day in range(0,5):
                week_days += summary_data[str(year)][str(month)][str(day)]
            week_days_perc = week_days/sum(week_days)

            month_data = np.append(week_days_perc, [label, year, month])

            train_data.append(month_data)
            # week_days_perc = pd.DataFrame(data=week_days_perc, columns=['exercise_start'])

    return summary_data, train_data


def generate_weekly_start_time_dict(activity_df, year):

    week_summary_data = {}

    # week_summary_data[str(year)] = {}

    years_data = activity_df[pd.DatetimeIndex(activity_df.start_date_local).year == year]

    for day in range(0,7):
        # print(f'\tProcessing Day {day}')

        week_summary_data[str(day)] = 24*[0]

        days_data = years_data[pd.DatetimeIndex(years_data.start_date_local).weekday == day]

        for hour in range(0,24):

            # print(f'\t\tAccumulating Hour {hour}')

            hours_data = days_data.set_index('start_date_local_raw')[pd.DatetimeIndex(days_data.start_date_local).hour == hour]

            week_summary_data[str(day)][hour] = len(hours_data)
    
    return week_summary_data
