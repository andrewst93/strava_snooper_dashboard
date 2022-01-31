import os
import time
import sys

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

src_path = os.path.abspath(os.path.join(".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from app import app
from pages.layouts import header, footer
from src.data.gcp_strava_data_load_preprocess import load_strava_activity_data_from_bq
from src.data.strava_data_load_preprocess import (
    load_strava_activity_data,
    load_employment_model_data,
    preprocess_employment_model_data,
)
from src.visualizations.dash_plots import (
    plot_lgbm_model_predictions,
    plot_logreg_model_predictions,
    plot_weekly_start_times,
    plot_eda_data,
    plot_training_data,
)

# used for debugging GCP app engine deployment issues
try:
    import googleclouddebugger

    googleclouddebugger.enable(breakpoint_enable_canary=True)
except ImportError:
    pass

# ----------- CONSTANTS -----------------------------------#

# What's the definition of "work hours", activities are grouped down 9-11 is any activities started 9:00 - 11:59
work_hours = [[9, 11], [13, 16]]

# ------------ END CONSTANTS -----------------------------#

# for real time running on GCP App engine pull data from BQ
raw_files_dict = load_strava_activity_data_from_bq()

# # Use following code to test with sample data locally
# # data_file_path = os.path.abspath(os.path.join(os.getcwd(), "data"))
# # print("Loading Strava Data: " + data_file_path)
# # # raw_files_dict = load_strava_activity_data(data_file_path)

num_activities = len(raw_files_dict["TyAndrews"].type)

activity_over_time = plot_eda_data(
    raw_files_dict["TyAndrews"], [2013, 2023], "distance_raw_km", "Month"
)

# # page intro card for top of page under header and link to blog post on it.
jumbotron = dbc.Jumbotron(
    [
        html.H6(
            "Millions of people upload their activities to Strava every day. This got me wondering...",
            className="display-6",
        ),
        html.Hr(className="my-2"),
        html.H5(
            "What does Strava know about us from our uploaded activities?",
            className="display-5",
        ),
        html.P(
            "For a full description of the analysis below see the full write up here."
        ),
        html.P(
            dbc.Button(
                "LEARN MORE",
                size="lg",
                href="https://ty-andrews.com/post/2021-02-23-what-does-strava-know-about-me/",
                color="primary",
                target="_blank",
            )
        ),  # , className="lead"
    ],
    className="dash-bootstrap",
)

activity_controls = html.Div(
    [
        html.P(
            "All of my past 8 years of activities uploaded to Strava were exported. "
            "From 2014-2017 I was training and going to school. 2017 to now I have been working "
            "full time."
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="eda-variable",
                    options=[
                        # {"label": col, "value": col} for col in raw_files_dict['TyAndrews'].columns
                        {"label": col[0], "value": col[1]}
                        for col in [
                            ["Activity Distance", "distance_raw_km"],
                            ["Elapsed Time", "elapsed_time_raw_hrs"],
                            ["Moving Time", "moving_time_raw_hrs"],
                            ["Elevation Gain", "elevation_gain"],
                        ]
                    ],
                    value="distance_raw_km",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Display Per:"),
                dcc.Dropdown(
                    id="eda-group",
                    options=[
                        # {"label": col, "value": col} for col in raw_files_dict['TyAndrews'].columns
                        {"label": col, "value": col}
                        for col in ["Day", "Week", "Month", "Quarter", "Year"]
                    ],
                    value="Month",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Year Range"),
                dcc.RangeSlider(
                    id="year-selector",
                    min=2013,
                    max=2023,
                    step=1,
                    value=[2013, 2023],
                    marks={
                        2013: "2013",
                        2014: "'14",
                        2015: "'15",
                        2016: "'16",
                        2017: "'17",
                        2018: "'18",
                        2019: "'19",
                        2020: "'20",
                        2021: "'21",
                        2022: "'22",
                        2023: "2023",
                    },
                ),
            ]
        ),
    ],
    # body=True,
)

# DCC card for showing my activitees over time etc.
data_intro_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("My Strava Data")),
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(activity_controls, width=3),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Loading(
                                    children=dcc.Graph(
                                        id="eda-plot", figure=activity_over_time
                                    ),
                                    type="cube",
                                    color="#e95420",
                                )
                            )
                        ),
                        width=9,
                    ),
                ]
            )
        ),
    ],
    color="#E0E0E0",
)

# pre-generate all plots for building th core components
unemployed_activity_start_time_fig = plot_weekly_start_times(
    raw_files_dict["TyAndrews"], 2014, 2017, work_hours, "Training + Studying"
)
employed_activity_start_time_fig = plot_weekly_start_times(
    raw_files_dict["TyAndrews"], 2018, 2020, work_hours, "Working Full Time"
)

train_employment_data, test_employment_data = load_employment_model_data()

train_data, train_labels = preprocess_employment_model_data(
    train_employment_data, work_hours
)
test_data, test_labels = preprocess_employment_model_data(
    test_employment_data, work_hours
)

training_data_plot = plot_training_data(
    train_data, train_labels, test_data, test_labels
)

lgbm_train_plot = plot_lgbm_model_predictions(
    train_data[["morn", "aft"]], train_data[["morn", "aft"]], train_labels, "train"
)  # optional: lgbm_model,
lgbm_test_plot = plot_lgbm_model_predictions(
    test_data[["morn", "aft"]], test_data[["morn", "aft"]], test_labels, "test"
)
logreg_train_plot = plot_logreg_model_predictions(
    train_data[["morn", "aft"]], train_labels, "train"
)  # Optional: logreg_model,
logreg_test_plot = plot_logreg_model_predictions(
    test_data[["morn", "aft"]], test_labels, "test"
)

# card holding the employment status hypothesis details and results plots.
employment_hypoth = dbc.Card(
    [
        dbc.CardHeader(html.H4("Does Strava know whether I'm employed?")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    "I wondered if Strava was able to determine which of their users are employed based on their workout habits. "
                                    "This could be useful to know for a number of reasons such as assessing potential subscribers or improving targeted marketing."
                                ),
                                html.Hr(),
                                html.H6("Hypothesis:"),
                                html.P(
                                    "If a users mid-week activities commonly start outside of work hours (9-12AM, 1-4PM) they are working."
                                ),
                                html.H6("First Look at the Data:"),
                                html.P(
                                    "From knowing what years I worked I can see that there is a significant trend that "
                                    " when I'm working fewer activities started during work hours."
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            html.H5("First Look at the Data"),
                                            justify="center",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="unempl-explore-plot",
                                                        figure=unemployed_activity_start_time_fig,
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    ),
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="empl-explore-plot",
                                                        figure=employed_activity_start_time_fig,
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    ),
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ),
                            width=9,
                        ),
                    ]
                ),
                html.Hr(className="my-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Training Data"),
                                html.P(
                                    "To apply a machine learning approach to this problem I tackled the data cleaning and prep. "
                                    "I decided to split the data into monthly data and create two features, % of weekday activities started during morning work hours, "
                                    " and percent started during afternoon hours. For more info on this decision see my blog post on it here:"
                                ),
                                html.P(
                                    dbc.Button(
                                        "BLOG POST",
                                        size="lg",
                                        href="https://ty-andrews.com/post/2021-02-23-what-does-strava-know-about-me/",
                                        color="primary",
                                        target="_blank",
                                    )
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="train-data-plot",
                                                        figure=training_data_plot,
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    ),
                                                    width=12,
                                                ),
                                                # dbc.Col(dcc.Graph(id='empl-explore-plot', figure=employed_activity_start_time_fig), width = 6)])
                                            ]
                                        )
                                    ]
                                )
                            ),
                            width=9,
                        ),
                    ]
                ),
                html.Hr(className="my-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Logistic Regression & LightGBM Models"),
                                html.P(
                                    "Taking into account the small data size a logistic regression model and Light Gradient Boosted Machine (LGBM) "
                                    "were trained on a 70/30 train test split. After some tuning the LGBM showed better accuracy but with more over fitting."
                                ),
                                html.Hr(),
                                html.H4("Conclusion"),
                                html.P(
                                    "These results are promising that with such a simple approach achieved results significantly better than a randomized model."
                                ),
                                html.Hr(),
                                html.H4("Future Improvements"),
                                html.P(
                                    "- as expected it's clear increasing the dataset size would help with reducing overfitting"
                                ),
                                html.P(
                                    "- outlier detection to remove months that don't have enough activities"
                                ),
                                html.P(
                                    "- including activities started in evenings, lunch times and mornings as features"
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        # dbc.Row(html.H5("Logistic Regression Model"), justify='center'),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Row(
                                                            html.H5(
                                                                "Training Data - 70%"
                                                            ),
                                                            justify="center",
                                                        ),
                                                        dcc.Graph(
                                                            id="logreg-train-data-plot",
                                                            figure=logreg_train_plot,
                                                            style={"height": "350px"},
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Row(
                                                            html.H5("Test Data - 30%"),
                                                            justify="center",
                                                        ),
                                                        dcc.Graph(
                                                            id="logreg-test-data-plot",
                                                            figure=logreg_test_plot,
                                                            style={"height": "350px"},
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            justify="center",
                                        ),
                                        # dbc.Row(html.H5("LightGBM Model"), justify='center'),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="lgbm-train-data-plot",
                                                        figure=lgbm_train_plot,
                                                        style={"height": "350px"},
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    ),
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id="lgbm-test-data-plot",
                                                        figure=lgbm_test_plot,
                                                        style={"height": "350px"},
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    ),
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ),
                            width=9,
                        ),
                    ]
                ),
            ]
        ),
    ],
    color="#E0E0E0",
)

# Actual page layout
layout = html.Div(
    [
        header(),
        jumbotron,
        dbc.Row(dbc.Col(data_intro_card, width={"size": 10, "offset": 1})),
        dbc.Row(dbc.Col(employment_hypoth, width={"size": 10, "offset": 1})),
        footer(),
    ],
)

# Updating the eda plot based on slider/dropdown selections
@app.callback(
    Output("eda-plot", "figure"),
    [
        Input("eda-variable", "value"),
        Input("year-selector", "value"),
        Input("eda-group", "value"),
    ],
)
def update_eda_plot(y_label, year_range, group_by):
    return plot_eda_data(raw_files_dict["TyAndrews"], year_range, y_label, group_by)
