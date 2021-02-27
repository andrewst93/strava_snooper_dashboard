import os
import glob
from datetime import datetime, timedelta
import time
import csv
import random

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression




curr_path = flask.helpers.get_root_path('dashboard')

# --------------- CONSTANTS & STYLING ---------------------
KAHA_Y1 = '#9D8C04'
KAHA_Y2 = '#B8A407'
KAHA_Y3 = '#D3BC00'
KAHA_Y4 = '#DBCC55'
KAHA_Y5 = '#F8E869'
KAHA_Y6 = '#FCEF88'

kaha_color_list = [KAHA_Y1, KAHA_Y2, KAHA_Y3, KAHA_Y4, KAHA_Y5, KAHA_Y6]

# colors = {
#     'background': '#FFFFFF',
#     'text': '#FFBE0B'
# }

pio.templates.default = "ggplot2"

#Style sheet setup
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [os.path.abspath(os.path.join(curr_path, 'assets', 'strava-analysis-style.css'))]
print(external_stylesheets)
# cmap = plt.get_cmap("tab10")
# ----- HELPER FUNCTIONS ------------------------------------

def preprocess_strava_df(raw_df):

    # Remove ectivites under 5 minutes in length
    processed_df = raw_df[(raw_df.elapsed_time_raw > 1200) & (raw_df.distance < 400)]
    print(f'\t{len(raw_df[(raw_df.elapsed_time_raw < 1200) & (raw_df.distance < 400)])} Activities Under 20min in Length, Removed from Dataset')

    processed_df = processed_df.convert_dtypes()
    processed_df[['distance','distance_raw']] = processed_df[['distance','distance_raw']].apply(pd.to_numeric)
    processed_df[['start_date_local']] = pd.to_datetime(processed_df['start_date_local_raw'], unit='s') #.apply(pd.to_datetime(unit='s'))
    processed_df['exer_start_time'] = pd.to_datetime(processed_df['start_date_local'].dt.strftime('1990:01:01:%H:%M:%S'), format='1990:01:01:%H:%M:%S')
    # processed_df['exer_start_time'] = pd.to_datetime(pd.to_datetime(processed_df['start_time']).dt.strftime('1990:01:01:%H:%M:%S'), format='1990:01:01:%H:%M:%S')
    processed_df['exer_start_time'] = processed_df['exer_start_time'].dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    processed_df['act_type_perc_time'] = (processed_df['moving_time_raw']/sum(processed_df['moving_time_raw']))

    return processed_df

def activity_pie_plot(df, val, label, title): #, text_pos
    
    fig = px.pie(df, 
                values=val, 
                names=label, 
                # color_discrete_sequence= kaha_color_list, 
                title=title,
                labels={'type':'Activity Type'}
            )
    fig.update_traces(textposition='inside', textinfo='label+percent', sort=False) 
    fig.update_layout(  showlegend=True, 
                        title_x=0.5, 
                        title_font_size=25,
                        font_size=15,
                        legend={'traceorder':'normal'}
                        )

    return fig

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
    
def generate_employment_prediction_model_data(activity_df, start_year, end_year, start_quarter, end_quarter, label):

    summary_data = {}

    train_data = []

    for year in range(start_year, end_year+1):

        summary_data[str(year)] = {}

        begin_quarter = 0
        stop_quarter = 4

        if year == start_year: begin_quarter = start_quarter
        if year == end_year: stop_quarter = end_quarter

        print(f'{year} {begin_quarter} {stop_quarter}')

        for quarter in range(begin_quarter, stop_quarter):

            summary_data[str(year)][str(quarter)] = {}

            print(f'\t{quarter}')

            for month in range(quarter*3, quarter*3+3):
                    
                print(f'\t\t{month}')

                for day in range(0,7):
                    # print(f'\tProcessing Day {day}')

                    summary_data[str(year)][str(quarter)][str(day)] = 24*[0]

                    # days_data = quarter_data[pd.DatetimeIndex(quarter_data.start_date_local).weekday == day]
                    
                    for hour in range(0,24):

                        # print(f'\t\tAccumulating Hour {hour}')

                        # hours_data = days_data.set_index('start_date_local')[pd.DatetimeIndex(days_data.start_date_local).hour == hour]
                        hours_data = activity_df[ (pd.DatetimeIndex(activity_df.start_date_local).year == year) & 
                                        (pd.DatetimeIndex(activity_df.start_date_local).month//3 == quarter) &
                                        (pd.DatetimeIndex(activity_df.start_date_local).weekday == day) &
                                        (pd.DatetimeIndex(activity_df.start_date_local).hour == hour)]

                        summary_data[str(year)][str(quarter)][str(day)][hour] = len(hours_data)

            week_days = np.array(24*[0])

            # Calculate what eprcentag of workout start times occur at each hour in the day.
            for day in range(0,5):
                week_days += summary_data[str(year)][str(quarter)][str(day)]
            week_days_perc = week_days/sum(week_days)

            quarter_data = np.append(week_days_perc, [label, year, quarter])

            train_data.append(quarter_data)
            # week_days_perc = pd.DataFrame(data=week_days_perc, columns=['exercise_start'])

    return summary_data, train_data

def plot_weekly_start_times(activity_df, start_year, end_year, title_descr):

    start_time_fig = go.Figure()

    work_hours = [[9,11],[13,16]]

    work_perc = 0

    for year in range(start_year, end_year + 1):

        weekly_start_times = generate_weekly_start_time_dict(activity_df, year)

        week_days = np.array(24*[0])
        weekend_days = np.array(24*[0])

        # Calculate what eprcentag of workout start times occur at each hour in the day.
        for day in range(0,5):
            week_days += weekly_start_times[str(day)]
        week_days_perc = week_days/sum(week_days)
        week_days_perc = pd.DataFrame(data=week_days_perc, columns=['exercise_start'])

        for time_span in work_hours:
            work_perc += sum(week_days_perc.exercise_start.iloc[time_span[0]-1:time_span[1]])
            start_time_fig.add_vrect(x0=time_span[0], x1=time_span[1]+1, line_width=0, fillcolor="orange", opacity=0.05)

        # if work_perc <= 0.3:
        #     work_status = 'Working'
        # else:
        #     work_status = 'Not Working'

        # process and get percentage of activities on weekend days over each hour
        for day in range(5,7):
            weekend_days += weekly_start_times[str(day)]

        weekend_days_perc = weekend_days/sum(weekend_days)
        weekend_days_perc = pd.DataFrame(data=weekend_days_perc, columns=['exercise_start'])

        print(f'Work Perc.: {work_perc:.3f}')

        start_time_fig.add_trace(go.Scatter(
                                        x=week_days_perc.index, 
                                        y=week_days_perc['exercise_start']*100,
                                        name=f'{year} Weekdays',
                                        mode='lines',
                                        line_color='#FFA400'
                                ))
        # start_time_fig.add_trace(go.Scatter(
        #                                 x=weekend_days_perc.index, 
        #                                 y=weekend_days_perc['exercise_start']*100,
        #                                 name=f'{year} Weekends',
        #                                 mode='lines',
        #                                 line_color='#009FFD'
        #                                 ))
    
    avg_work_percent = work_perc/(end_year - start_year + 1)

    start_time_fig.update_layout(
        title=f'{start_year} - {end_year}: {title_descr}<br>{avg_work_percent*100:.0f}% of Weekday Activities Started During Work Hours',
        xaxis=dict(title="Activity Start Time (24hr time)"),
        yaxis=dict(title="Percentage of Activities", ticksuffix='%'),
        title_x=0.5, 
        title_font_size=25,
        font_size=15,
    )

    return start_time_fig

def load_employment_model_data():

    model_data_file_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    print('Loading Employment Model Data: ' + data_file_path)
    start = time.process_time()

    train_data = os.path.join(data_file_path, "train_employment_data.csv")
    test_data = os.path.join(data_file_path, "test_employment_data.csv")

    files = [train_data, test_data]
    all_data = []
    for f in files:
        data = pd.read_csv(f)

        all_data.append(data)

    return all_data[0], all_data[1]

def plot_logreg_model_predictions(logreg_model, data, labels):

    prediction = logreg_model.predict(data)

    acc = logreg_model.score(data, labels)
    print(f'Accuracy: {acc:.3f}')

    #Create logistic regression heat map
    h=0.001
    x_min, x_max = data['morn'].min() - 0.02, data['morn'].max() + 0.02
    y_min, y_max = data['aft'].min() - 0.02, data['aft'].max() + 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
                        
    y_ = np.arange(y_min, y_max, h)

    Z = logreg_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,:1]
    Z = Z.reshape(xx.shape)

    data_plot = px.scatter(   x=data['morn'], y=data['aft'], 
                                    color=labels, color_discrete_map={'unemployed':'#009FFD', 'employed': '#FFA400',}, 
                                    symbol= np.where(prediction == np.array(labels), 'Correct', 'Incorrect'),
                                    symbol_map= {'Correct':'circle',
                                                'Incorrect': 'x'},
                                    labels={
                                        'color':'Label',
                                        'symbol': 'Model Prediction'
                                    }
                                ).update_traces(marker = dict(size = 15, 
                                line_width=2, line_color='black'))
    data_plot.update_layout(
        title=f'Employment Predictions from Weekday Activities Started During Work Hours<br>Model Accuracy: {acc*100:.0f}%',
        xaxis=dict(title="Activities During Morning Work Hours", tickformat=',.0%', range=[x_min,x_max]),
        yaxis=dict(title="Activities During Afternoon Work Hours", tickformat=',.0%', range=[y_min,y_max]),
        title_x=0.5, 
        title_font_size=15,
        font_size=10,
    )

    data_plot.add_trace(go.Heatmap(x=xx[0], y=y_, z=Z,
                    colorscale= [[0, '#009FFD'], [0.45, '#009FFD'], [0.55,'#FFA400'], [1,'#FFA400']],
                    opacity=0.3,
                    showscale=False,)
    )

    return data_plot

# --------------------- END HELPER FUNCTIONS -------------------------------------------

# Load sample data
data_file_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
print('Loading Data: ' + data_file_path)
start = time.process_time()
raw_files_dict = {}
for f in glob.glob(os.path.join(data_file_path, "*StravaData*.csv")):
    user = os.path.basename(f).split('_')[0]
    print(f'{user} Data Found')
    raw_files_dict[user] = preprocess_strava_df(pd.read_csv(f))
print(f'\tTook {time.process_time()- start:.2f}s')

num_activities = len(raw_files_dict['TyAndrews'].type)

print('Generating Pie Plot and Scatter.')
start = time.process_time()
activity_type_pie_plot = activity_pie_plot(
    df=raw_files_dict['TyAndrews'],
    val="act_type_perc_time",
    label='type',
    title=f'Activity Distribution: {num_activities} Total Activities'
)

activity_over_time = px.scatter(raw_files_dict['TyAndrews'], 
                                x="start_time", 
                                y="distance", 
                                title='Activity Distance Over Time', 
                                color='type',
                                labels={
                                    'distance': 'Distance (km)',
                                    'type': "Activity Type",
                                    'start_time': 'Date'
                                })
activity_over_time.update_layout(showlegend=True, 
                        title_x=0.5, 
                        title_font_size=25,
                        font_size=15,
                        legend={'traceorder':'normal'}
                        )
print(f'\tTook {time.process_time()- start:.2f}s')

print('Generating Weekly Start Plots.')
start = time.process_time()
unemployed_activity_start_time_fig = plot_weekly_start_times(raw_files_dict['TyAndrews'], 2014, 2017, 'Training + Studying')
employed_activity_start_time_fig = plot_weekly_start_times(raw_files_dict['TyAndrews'], 2018, 2020, 'Working Full Time')
print(f'\tTook {time.process_time()- start:.2f}s')

try: 
    train_employment_data, test_employment_data = load_employment_model_data()

except FileNotFoundError:

    print('Generating Train Data.')
    start = time.process_time()
    summary_data, unemployed_train = generate_employment_prediction_model_data(raw_files_dict['TyAndrews'], 2014, 2017, 0,4, 'unemployed')
    summary_data, employed_train = generate_employment_prediction_model_data(raw_files_dict['TyAndrews'], 2018, 2020, 0,4, 'employed')
    print(f'\tTook {time.process_time()- start:.2f}s')

    # Shuffle employed and unemployed data, seed for consistency
    random.Random(4).shuffle(unemployed_train)
    random.Random(4).shuffle(employed_train)

    train_test_split = 0.7
    empl_split = round(train_test_split*len(employed_train))
    unempl_split = round(train_test_split*len(unemployed_train))
    print(f'Train test Split: {train_test_split:.2f} Empl. Train Size: {empl_split} Unempl. Train Size: {unempl_split}')
    columns = ['0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23',
                'label', 'year', 'quarter']
    train_employment_data   = pd.DataFrame(unemployed_train[0:unempl_split] + employed_train[0:empl_split], columns=columns) 
    test_employment_data    = pd.DataFrame(unemployed_train[unempl_split:] + employed_train[empl_split:], columns=columns)

    train_employment_data.to_csv(os.path.join('data','train_employment_data.csv'), index=False)
    test_employment_data.to_csv(os.path.join('data','test_employment_data.csv'), index=False)


train_data = train_employment_data.iloc[:,0:24]
train_labels = train_employment_data['label']
test_data = test_employment_data.iloc[:,0:24]
test_labels = test_employment_data['label']

morning = [9,11]
afternoon = [13,16]

train_data['morn'] = train_data.iloc[:, morning[0]-1:morning[1]].sum(axis=1)
train_data['aft'] = train_data.iloc[:, afternoon[0]-1:afternoon[1]].sum(axis=1)

test_data['morn'] = test_data.iloc[:, morning[0]-1:morning[1]].sum(axis=1)
test_data['aft'] = test_data.iloc[:, afternoon[0]-1:afternoon[1]].sum(axis=1)

print("Training LogReg Model")
logreg_model = LogisticRegression(
    penalty='l2',
    C=1000,
    solver='lbfgs'
)

logreg_model.fit(train_data[['morn', 'aft']], train_labels)

train_data_plot = plot_logreg_model_predictions(logreg_model, train_data[['morn','aft']], train_labels)
test_data_plot = plot_logreg_model_predictions(logreg_model, test_data[['morn','aft']], test_labels)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style = {'width': '100%', 'display': 'inline-block', 'border-radius': 15},  

    children=[
        html.Div( className= 'one-third column', style={'textAlign': 'center', 'verticalAlign': 'bottom'},
            children=[
                    html.H1(
                        children = 'Strava Snooper',
                            style={  
                                'display': 'inline-block',
                                # 'verticalAlign': 'bottom',
                                'marginLeft': '1.0rem'
                            }
                        ), 
                    ]
                ),
        html.Div(className='offset by one-third columns one-third column', style={'textAlign': 'center', 'verticalAlign': 'middle',},
            children=[
                html.H4(
                    children = 'What does Strava know about you?',
                        style={  
                            'display': 'inline-block',
                            'verticalAlign': 'middle',
                            'marginLeft': '1.0rem',
                            'marginTop' : '2.0rem',
                            'fontFamily': 'Bungee'
                        }
                ), 
            ]
        ),
        html.Div(className='offset by two-third columns one-third column', style={'textAlign': 'right', 'verticalAlign':'middle'},
            children=[
                html.H5(
                    children = 'Created by: Ty Andrews',
                        style={  
                            'display': 'inline-block',
                            'verticalAlign': 'middle',
                            # 'marginLeft': '1.0rem',
                            'fontFamily': 'Bungee'
                        }
                ), 
                html.Div(
                    children = [html.A("ty-andrews.com", href='https://ty-andrews.com/', target="_blank", style={'fontFamily':'Bungee', 'fontSize':'2.6rem'})]
                )
                
            ]
        ),

        html.Div(className='pretty_container twelve columns', 

            children=[
                html.Div(className='twelve columns',
                    children = [  
                        dcc.Markdown(children = '''If you\'re unfamiliar with the world of activity monitoring, **Strava services over 55 million users** 
                                    for being a training and virtual competition platform. Athletes from around the world run, ride and 
                                    swim along the same routes as one another and their times for those segments are compared on a leaderboard. 
                                    In recent years Strava has grown to an **annual revenue of over 72 million dollars** from its Premium subscription service 
                                     for tracking and monitoring performance.''',
                                    style = { 'font-size': '2.5rem', 'line-height': '1.6',  'letter-spacing': '0', 'margin-bottom':' 0.75rem', 'margin-top': '0.75rem'},                      
                        ),
                        html.Div(className='twelve columns',style = {'textAlign': 'center','verticalAlign': 'center'},
                            children = [
                                html.H4(
                                    children = "Strava's users statistics are impressive",
                                    style = {'marginBottom': '3.0rem', 'marginTop': '2.0rem', 'verticalAlign': 'bottom'}
                                )
                            ]
                        ),
                        html.Div(className='six columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H3(
                                    children = "3 billion",
                                    style = {'marginBottom': '3.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                ),
                                html.H3(
                                    children = '11.2 billion' ,
                                    style = {'marginBottom': '3.0rem', 'marginTop': '1.5rem', 'fontWeight': 'bold'}
                                ),
                                html.H3(
                                    children = '80%' ,
                                    style = {'marginBottom': '3.0rem', 'marginTop': '1.5rem', 'fontWeight': 'bold'}
                                ),
                                html.H3(
                                    children = '20' ,
                                    style = {'marginBottom': '3.0rem', 'marginTop': '1.5rem', 'fontWeight': 'bold'}
                                ),
                            ]
                        ),
                        html.Div(className='offset by six columns six columns', style={'textAlign': 'left' },
                            children=[
                                html.H4(
                                    children = "Activities Uploaded in 2020",
                                    style = {'marginBottom': '3.5rem', 'marginTop': '2.5rem'}
                                ),
                                html.H4(
                                    children = "Kilometers recorded in 2020",
                                    style = {'marginBottom': '3.5rem', 'marginTop': '2.5rem'}
                                ),
                                html.H4(
                                    children = "Of users are outside the US",
                                    style = {'marginBottom': '3.5rem', 'marginTop': '2.5rem'}
                                ),
                                html.H4(
                                    children = "Activities uploaded every second",
                                    style = {'marginBottom': '3.5rem', 'marginTop': '2.5rem'}
                                ),
                            ]
                        ),
                        html.Div(className='twelve columns', style={'textAlign': 'center' },
                            children=[
                                html.H4(
                                    children = 'This got me wondering...' ,
                                    style = {'marginBottom': '3.0rem', 'marginTop': '1.5rem', 'fontWeight': 'bold'}
                                ),
                                html.H3(
                                    children = 'What does Strava know about me from my uploaded activities?' ,
                                    style = {'marginBottom': '3.0rem', 'marginTop': '1.5rem', 'fontWeight': 'bold'}
                                ),
                            ]
                        )
                    ]
                )
            ]
        ), 
        html.Div(className='pretty_container twelve columns', style={'textAlign': 'center'},
            children=[
                html.H3(
                    children = 'The Data',
                    style={  
                        'display': 'inline-block',
                        'fontFamily': 'Bungee'
                    }
                ), 
                dcc.Markdown(children = 
'''To get your data out of Strava I found two options.   

&nbsp  
1. A bulk export of all the raw GPS and .fit files along with other basic information which is nicely provided by Strava.  
2. Manually exporting the summary of exercises one page at a time in JSON format.  

&nbsp  
For this initial analysis I only wanted high level activity info (activity type, start times, length, distance etc.).  
Luckily I found this slick post from Scott Dawson, a UX Designer from New York, who figured out how to get all activity info
in a single export: [How to Export Strava Workout Data](https://scottpdawson.com/export-strava-workout-data/).  

&nbsp;  
This results in a CSV with the following information:''',
                    style = { 'font-size': '2.5rem', 'line-height': '1.6',  'letter-spacing': '0', 'margin-bottom':' 0.75rem', 'margin-top': '0.75rem'},                      
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Activity Name",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "Chosen name for the activity or defaults of \"Morning Ride\" \"Afternoon Run\" etc.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Activity Type",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "The sport or activity type e.g. Run, Weights, Hike, etc.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Gear ID",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "The gear you used and entered into Strava for tracking wear etc.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Start Date & Time",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "Pretty self explanatory!",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Distance",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "Total distance of the activity",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Moving & Elapsed Time",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "Moving time is determined by Strava based on GPS or other speed sources, elapsed time is total recording length",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H5(
                                    children = "Elevation & Calories",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H6(
                                    children = "The total elevation that was gained and calories burned, don't know exact method for the calorie estimation.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className= 'pretty_container six columns',style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                    children=[
                        dcc.Graph(
                            figure = activity_type_pie_plot, 
                            id= 'act_type_pie',
                            style= {'width': '100%', 
                                    'height': '450px',
                                    'display':'inline-block'},
                        )
                    ]
                ),
                html.Div(className= 'pretty_container offset by six columns six columns',style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' , 'display':'inline-block'},
                    children=[
                        dcc.Graph(
                            figure = activity_over_time, 
                            id= 'act_type_pie2',
                            style= {'width': '100%', 
                                    'height': '450px',
                                    # 'display':'inline-block'
                                    },
                        )
                    ]
                )
            ]
        ),
        html.Div(className='pretty_container twelve columns', style={'textAlign': 'center'},
            children=[
                html.H3(
                    children = 'Predicting if Users Are Employed',
                    style={  
                        'display': 'inline-block',
                        'fontFamily': 'Bungee'
                    }
                ), 
                dcc.Markdown(children = '''
I wondered if Strava was able to determine which of their users are employed. This could be extremely useful in evaluating
the proportion of their user base that have higher disposable income and thus most likely to subscribe to the Premium service.  

&nbsp  
To try and see this, I know that in my data from 2014 - 2018 I was in school with a flexible schedule and training semi-proffessionally. 
In 2018 I graduated with my degree with Honors in Mechatronics Engineering from the University of Victoria and began my career at the sports tech 
and IoT company [4iiii Innovations Inc.](https://4iiii.com/)  

&nbsp  
Therefore, I should be able to see a significant difference in my activity's between those two time periods.
''',
                    style = { 'font-size': '2.5rem', 'line-height': '1.6',  'letter-spacing': '0', 'margin-bottom':' 0.75rem', 'margin-top': '0.75rem'},
                ),
                html.Div(className='twelve columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                    children=[
                        html.Div(className='four columns',style = {'textAlign': 'right','verticalAlign': 'middle'},
                            children=[
                                html.H4(
                                    children = "Hypothesis",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.0rem', 'fontWeight': 'bold'}
                                )
                            ]
                        ),
                        html.Div(className='offset by five columns seven columns', style={'textAlign': 'left' },
                            children=[
                                html.H5(
                                    children = "If a users mid-week activities commonly start outside of work hours (9-12AM, 1-4PM) they are working.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(className= 'pretty_container twelve columns',style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                    children=[
                        html.Div(className= 'twelve columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center' },
                            children = [
                                html.H4(
                                    children = 'Initial Exploration',
                                    style={  
                                        'display': 'inline-block',
                                        'fontFamily': 'Bungee'
                                    }
                                ),   
                            ]
                        ),
                        html.Div(className= 'eight columns offset-by-two column', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center'}, #
                            children = [
                                html.H5(
                                    children = "To see if this was realistic I grouped each year's weekday activities by start time. "
                                    " Converting those into a percentage of total activities to account for weeks with varying "
                                    "numbers of activities.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                ),
                            ]
                        ),
                        html.Div(className= 'six columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                            children = [
                                dcc.Graph(
                                    figure = unemployed_activity_start_time_fig, 
                                    id= 'unemployed_weekly_start',
                                    style= {'width': '100%', 
                                            'height': '450px',
                                            'display':'inline-block'},
                                )
                            ]
                        ),
                        html.Div(className= 'six columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                            children = [
                                dcc.Graph(
                                    figure = employed_activity_start_time_fig, 
                                    id= 'employed_weekly_start',
                                    style= {'width': '100%', 
                                            'height': '450px',
                                            'display':'inline-block'},
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(className= 'pretty_container twelve columns',style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                    children=[
                        html.Div(className= 'twelve columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center' },
                            children = [
                                html.H4(
                                    children = 'Building the Model',
                                    style={  
                                        'display': 'inline-block',
                                        'fontFamily': 'Bungee'
                                    }
                                ),   
                            ]
                        ),
                        html.Div(className= 'eight columns offset-by-two column', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center'}, #
                            children = [
                                html.H5(
                                    children = "To build a predictive model I first thought of the time frame over which Strava might want this information. "
                                    "Quarterly seemed like a reasonable starting point to balance having enough data while being able to make actionable "
                                    "changes to marketing plans, future earning reports etc.",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem'}
                                ),
                            ]
                        ),
                        html.Div(className= 'eight columns offset-by-two column', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center'}, #
                            children = [
                                html.H5(
                                    children = "Model Selection: Logistic Regression",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem','fontFamily': 'Bungee'}
                                ),
                            ]
                        ),
                        html.Div(className= 'eight columns offset-by-two column', style={'backgroundColor':'#FFFFFF', 'textAlign': 'center'}, #
                            children = [
                                html.H5(
                                    children = ["The main factors considered for model selection is the small data set available for testing (~20 quarters of data). "
                                    "As is common with small datasets overfitting will be an issue. The dataset is also unbalanced (60/40 split for binary classification). ",
                                    "Logistic Regression was chosen as the starting point as it's simple to train, simple to regularize and quick to train.",
                                    ]
                                ),
                                html.H5(
                                    children = "Feature Selection",
                                    style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem','fontFamily': 'Bungee'}
                                ),
                                html.H5(
                                    children = "Feature selection for Logistic Regression is susceptible to the 1-in-10 rule of thumb in which the number of features should be "
                                    "approximately 10 percent of the positive labelled data (employed). Since I have ~10 employed samples I'm going to break the rule "
                                    "and have 2 features, percentage of activities during morning work hours (9-12AM) and afternoon work hours (1-4PM)."
                                ),
                            ]
                        ),
                        html.Div(className= 'twelve columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                            children = [
                                html.Div(className= 'six columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                                    children = [
                                        html.H5(
                                            children = "Training Data - 70%",
                                            style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem','fontFamily': 'Bungee'}
                                        ),
                                        dcc.Graph(
                                            figure = train_data_plot, 
                                            id= 'train_data_plot',
                                            style= {'width': '100%', 
                                                    'height': '550px',
                                                    'display':'inline-block'},
                                        )
                                    ]
                                ),
                                html.Div(className= 'six columns', style={'backgroundColor':'#FFFFFF', 'textAlign': 'left' },
                                    children = [
                                        html.H5(
                                            children = "Test Data - 30%",
                                            style = {'marginBottom': '1.0rem', 'marginTop': '2.25rem','fontFamily': 'Bungee'}
                                        ),
                                        dcc.Graph(
                                            figure = test_data_plot, 
                                            id= 'test_data_plot',
                                            style= {'width': '100%', 
                                                    'height': '550px',
                                                    'display':'inline-block'},
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True, use_reloader=True) #dev_tools_hot_reload=False, 