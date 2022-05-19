import os
import time
import sys

import dash
from dash import dcc, html, callback_context, no_update
from dash.exceptions import PreventUpdate
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
from src.visualizations.employment_dash_plots import (
    plot_lgbm_model_predictions,
    plot_logreg_model_predictions,
    plot_weekly_start_times,
    plot_eda_data,
    plot_training_data,
)
from src.visualizations import kudos_prediction_dash_plots
from src.models.predict_kudos import predict_kudos

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

# def get_layout():

# for real time running on GCP App engine pull data from BQ

# tmp = load_strava_activity_data_from_bq()
# strava_df = tmp["TyAndrews"]

# recent_rides_df = strava_df[
#     (strava_df.type == "Ride")
#     & (strava_df.distance_raw_km > 10)
#     & (strava_df.achievement_count < 60)
# ].iloc[0:200]

# pred_num_kudos, pred_perc_kudos = predict_kudos(
#     custom_name=recent_rides_df.custom_name_bool.tolist(),
#     distance=recent_rides_df.distance_raw_km.astype(int).tolist(),
#     achievements=recent_rides_df.achievement_count.astype(int).tolist(),
#     elevation=recent_rides_df.elevation_gain.astype(int).tolist(),
#     num_followers=138,
# )

# actual_num_kudos = recent_rides_df.kudos.tolist()

# kudos_actuals_plot = kudos_prediction_dash_plots.plot_prediction_vs_actual_data(
#     actual_kudos=actual_num_kudos, pred_kudos=pred_num_kudos
# )

# activity_over_time = plot_eda_data(strava_df, [2013, 2023], "distance_raw_km", "Month")

# train_employment_data, test_employment_data = load_employment_model_data()

# train_data, train_labels = preprocess_employment_model_data(
#     train_employment_data, work_hours
# )
# test_data, test_labels = preprocess_employment_model_data(
#     test_employment_data, work_hours
# )

# training_data_plot = plot_training_data(
#     train_data, train_labels, test_data, test_labels
# )

# # page intro card for top of page under header and link to blog post on it.
jumbotron = dbc.Container(
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
                href="https://ty-andrews.com",
                color="primary",
                target="_blank",
            )
        ),  # , className="lead"
    ],
    className="h-100 p-5 bg-light border rounded-3",  # "dash-bootstrap py-3",
    fluid=True,
)

activity_controls = dbc.Col(
    [
        html.P(
            "All of my past 8 years of activities uploaded to Strava were exported. "
            "From 2014-2017 I was training and going to school. 2017 to now I have been working "
            "full time."
        ),
        dbc.Row(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="eda-variable",
                    options=[
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
        dbc.Row(
            [
                dbc.Label("Display Per:"),
                dcc.Dropdown(
                    id="eda-group",
                    options=[
                        {"label": col, "value": col}
                        for col in ["Day", "Week", "Month", "Quarter", "Year"]
                    ],
                    value="Month",
                ),
            ]
        ),
        dbc.Row(
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

# DCC card for showing my activities over time etc.
data_intro_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Explore My Strava Data")),
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(activity_controls, width=12, lg=3),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Loading(
                                    children=dcc.Graph(
                                        id="eda-plot",
                                        figure={},  # activity_over_time,
                                        config={"displayModeBar": False},
                                    ),
                                    type="cube",
                                    color="#e95420",
                                )
                            )
                        ),
                        width=12,
                        lg=9,
                    ),
                ]
            )
        ),
    ],
    color="#E0E0E0",
)


kudos_prediction = dbc.Card(
    [
        dbc.CardHeader(html.H4("How can you maximize kudos on your next ride?")),
        dbc.CardBody(
            [
                html.P(
                    "With social media likes and views becoming increasingly monetized I wondered if Strava would see a similar drive for maximizing engagement and kudos. "
                    "To be able to predict how many kudos you'd get based on your plan gives the opportunity to 'maximize' your kudos with "
                    "just a couple more km's or meters of elevation climbed."
                ),
                html.Div(
                    [
                        dbc.Button(
                            "FIND OUT HOW MANY KUDOS YOU'LL GET",
                            href="/pages/kudos-prediction",
                            color="primary",
                            size="lg",
                            style={"font-size": 17},
                        ),
                    ],
                    className="d-grid gap-2 col-10 mx-auto",
                ),
                dbc.Card(
                    dbc.CardBody(
                        children=dcc.Graph(
                            id="kudos-actual-plot",
                            figure={},  # kudos_actuals_plot,
                            config={"displayModeBar": False},
                        ),
                    )
                ),
            ]
        ),
    ],
    color="#E0E0E0",
)

employment_prediction = dbc.Card(
    [
        dbc.CardHeader(html.H4("Could Strava know if you're employed?")),
        dbc.CardBody(
            [
                html.P(
                    "I wondered if Strava could infer who's employed vs. not employed to better target promotions as well as estimate how much of their user base "
                    "might have extra cash to pay for a subscription."
                ),
                html.Div(
                    [
                        dbc.Button(
                            "CLICK HERE TO FIND OUT",
                            href="/pages/employment-prediction",
                            color="primary",
                            size="lg",
                            style={"font-size": 17},
                        ),
                    ],
                    className="d-grid gap-2 col-10 mx-auto",
                ),
                dbc.Card(
                    dbc.CardBody(
                        children=dcc.Graph(
                            id="empl-train-data",
                            figure={},  # training_data_plot,
                            config={"displayModeBar": False},
                        ),
                    )
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
        dbc.Row(
            [
                dbc.Col(
                    kudos_prediction,
                    width={"size": 12, "offset": 0},
                    lg={"size": 5, "offset": 1},
                ),
                dbc.Col(
                    employment_prediction,
                    width={"size": 12, "offset": 0},
                    lg={"size": 5, "offset": 0},
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                data_intro_card,
                width={"size": 12, "offset": 0},
                lg={"size": 10, "offset": 1},
            ),
        ),
        footer(),
        html.Div(id="page-load-div"),
    ],
)

# return layout


# initial page load callback
@app.callback(
    Output("kudos-actual-plot", "figure"),
    Output("empl-train-data", "figure"),
    Output("eda-plot", "figure"),
    Output("strava-data", "data"),
    Output("empl-data", "data"),
    Output("kudos-actuals", "data"),
    Input("eda-variable", "value"),
    Input("year-selector", "value"),
    Input("eda-group", "value"),
    Input("page-load-div", "children"),
    [
        State("strava-data", "data"),
        State("empl-data", "data"),
        State("kudos-actuals", "data"),
    ],
    prevent_initial_call=True,
)
def display_page(
    y_label, year_range, group_by, _page_load, strava_dict, empl_dict, kudos_dict
):

    if strava_dict is None:
        strava_df = load_strava_activity_data_from_bq()["TyAndrews"]
        strava_dict = strava_df.to_dict("records")
    else:
        strava_df = pd.DataFrame(strava_dict)

    if kudos_dict is None:
        recent_rides_df = strava_df[
            (strava_df.type == "Ride")
            & (strava_df.distance_raw_km > 10)
            & (strava_df.achievement_count < 60)
        ].iloc[0:200]

        actual_num_kudos = recent_rides_df.kudos.tolist()

        pred_num_kudos, pred_perc_kudos = predict_kudos(
            custom_name=recent_rides_df.custom_name_bool.tolist(),
            distance=recent_rides_df.distance_raw_km.astype(int).tolist(),
            achievements=recent_rides_df.achievement_count.astype(int).tolist(),
            elevation=recent_rides_df.elevation_gain.astype(int).tolist(),
            num_followers=138,
        )

        kudos_dict = {
            "pred_num_kudos": pred_num_kudos,
            "actual_num_kudos": actual_num_kudos,
        }

    else:
        pred_num_kudos = kudos_dict["pred_num_kudos"]
        actual_num_kudos = kudos_dict["actual_num_kudos"]

    if empl_dict is None:
        train_employment_data, test_employment_data = load_employment_model_data()

        train_data, train_labels = preprocess_employment_model_data(
            train_employment_data, work_hours
        )
        test_data, test_labels = preprocess_employment_model_data(
            test_employment_data, work_hours
        )

        empl_dict = {
            "train_data": train_data.to_dict(),
            "train_labels": train_labels.to_dict(),
            "test_data": test_data.to_dict(),
            "test_labels": test_labels.to_dict(),
        }
    else:
        train_data = pd.DataFrame(empl_dict["train_data"])
        train_labels = pd.Series(empl_dict["train_labels"])
        test_data = pd.DataFrame(empl_dict["test_data"])
        test_labels = pd.Series(empl_dict["test_labels"])

    inputs = []
    for ctx in callback_context.triggered:
        print(ctx["prop_id"])
        inputs.append(ctx["prop_id"])

    # initial page load only
    if "page-load-div.children" in inputs:
        print("[homepage} Initial load of plots.")
        activity_over_time = plot_eda_data(strava_df, year_range, y_label, group_by)

        training_data_plot = plot_training_data(
            train_data, train_labels, test_data, test_labels
        )

        kudos_actuals_plot = kudos_prediction_dash_plots.plot_prediction_vs_actual_data(
            actual_kudos=actual_num_kudos, pred_kudos=pred_num_kudos
        )

        return (
            kudos_actuals_plot,
            training_data_plot,
            activity_over_time,
            strava_dict,
            empl_dict,
            kudos_dict,
        )
    elif (
        ("eda-variable.value" in inputs)
        or ("year-selector.value" in inputs)
        or ("eda-group.value" in inputs)
    ):
        print("[homepage} Update activity plot only.")
        activity_over_time = plot_eda_data(strava_df, year_range, y_label, group_by)

        return (
            no_update,
            no_update,
            activity_over_time,
            strava_dict,
            empl_dict,
            kudos_dict,
        )
    else:
        return no_update, no_update, no_update, strava_dict, empl_dict, kudos_dict


# Updating the eda plot based on slider/dropdown selections
# @app.callback(
#     [Output("eda-plot", "figure")],  # , Output("strava-data", "data")
#     [
#         Input("eda-variable", "value"),
#         Input("year-selector", "value"),
#         Input("eda-group", "value"),
#     ],
#     State("strava-data", "data"),
# )
# def update_eda_plot(y_label, year_range, group_by, strava_dict):

#     if strava_dict is None:
#         strava_df = load_strava_activity_data_from_bq()["TyAndrews"]
#     else:
#         strava_df = pd.DataFrame(strava_dict)

#     return plot_eda_data(strava_df, year_range, y_label, group_by)
