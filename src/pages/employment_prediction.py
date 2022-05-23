import os
import time
import sys

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_loading_spinners
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
from src.pages.layouts import header, footer, blank_placeholder_plot
from src.data.gcp_strava_data_load_preprocess import load_strava_activity_data_from_bq
from src.data.strava_data_load_preprocess import (
    load_employment_model_data,
    preprocess_employment_model_data,
)
from src.visualizations.employment_dash_plots import (
    plot_lgbm_model_predictions,
    plot_logreg_model_predictions,
    plot_weekly_start_times,
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

loading_speed_multiplier = 1.5
loading_color = "#e95420"
loading_width = 125


# ------------ END CONSTANTS -----------------------------#


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
                            width=12,
                            lg=3,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Center(
                                            html.H5("First Look at the Data"),
                                            # justify="end",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dash_loading_spinners.Ring(
                                                        children=dcc.Graph(
                                                            id="unempl-explore-plot",
                                                            figure=blank_placeholder_plot(),
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                        speed_multiplier=loading_speed_multiplier,
                                                        width=loading_width,
                                                        color=loading_color,
                                                    ),
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    dash_loading_spinners.Ring(
                                                        children=dcc.Graph(
                                                            id="empl-explore-plot",
                                                            figure=blank_placeholder_plot(),
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                        speed_multiplier=loading_speed_multiplier,
                                                        width=loading_width,
                                                        color=loading_color,
                                                    ),
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ),
                            width=12,
                            lg=9,
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
                            width=12,
                            lg=3,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dash_loading_spinners.Ring(
                                                        dcc.Graph(
                                                            id="train-data-plot",
                                                            figure=blank_placeholder_plot(),
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                        speed_multiplier=loading_speed_multiplier,
                                                        width=loading_width,
                                                        color=loading_color,
                                                    ),
                                                    width=12,
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ),
                            width=12,
                            lg=9,
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
                            width=12,
                            lg=3,
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
                                                        html.Center(
                                                            html.H5(
                                                                "Training Data - 70%"
                                                            ),
                                                        ),
                                                        dash_loading_spinners.Ring(
                                                            dcc.Graph(
                                                                id="logreg-train-data-plot",
                                                                figure=blank_placeholder_plot(),
                                                                style={
                                                                    "height": "350px"
                                                                },
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            ),
                                                            speed_multiplier=loading_speed_multiplier,
                                                            width=loading_width,
                                                            color=loading_color,
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Center(
                                                            html.H5("Test Data - 30%"),
                                                        ),
                                                        dash_loading_spinners.Ring(
                                                            children=dcc.Graph(
                                                                id="logreg-test-data-plot",
                                                                figure=blank_placeholder_plot(),
                                                                style={
                                                                    "height": "350px"
                                                                },
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            ),
                                                            speed_multiplier=loading_speed_multiplier,
                                                            width=loading_width,
                                                            color=loading_color,
                                                        ),
                                                    ],
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ],
                                            justify="center",
                                        ),
                                        # dbc.Row(html.H5("LightGBM Model"), justify='center'),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dash_loading_spinners.Ring(
                                                        children=dcc.Graph(
                                                            id="lgbm-train-data-plot",
                                                            figure=blank_placeholder_plot(),
                                                            style={"height": "350px"},
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                        speed_multiplier=loading_speed_multiplier,
                                                        width=loading_width,
                                                        color=loading_color,
                                                    ),
                                                    width=12,
                                                    lg=6,
                                                ),
                                                dbc.Col(
                                                    dash_loading_spinners.Ring(
                                                        children=dcc.Graph(
                                                            id="lgbm-test-data-plot",
                                                            figure=blank_placeholder_plot(),
                                                            style={"height": "350px"},
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                        speed_multiplier=loading_speed_multiplier,
                                                        width=loading_width,
                                                        color=loading_color,
                                                    ),
                                                    width=12,
                                                    lg=6,
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ),
                            width=12,
                            lg=9,
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
        dbc.Row(
            dbc.Col(
                employment_hypoth,
                width={"size": 12, "offset": 0},
                lg={"size": 10, "offset": 1},
            )
        ),
        footer(),
        html.Div(id="employ-page-load-div"),
    ],
)

# On page load, populate first necessary data/plots
@app.callback(
    Output("unempl-explore-plot", "figure"),
    Output("empl-explore-plot", "figure"),
    Output("train-data-plot", "figure"),
    Output("logreg-train-data-plot", "figure"),
    Output("logreg-test-data-plot", "figure"),
    Output("lgbm-test-data-plot", "figure"),
    Output("lgbm-train-data-plot", "figure"),
    Input("employ-page-load-div", "children"),
    [
        State("strava-data", "data"),
        State("empl-data", "data"),
    ],
    prevent_initial_call=False,
)
def load_app_data(_page_load, strava_dict, empl_dict):

    if strava_dict is None:
        strava_df = load_strava_activity_data_from_bq()["TyAndrews"]
        strava_dict = strava_df.to_dict("records")
    else:
        strava_df = pd.DataFrame(strava_dict)

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

    train_data = pd.DataFrame(empl_dict["train_data"])
    train_labels = pd.Series(empl_dict["train_labels"])
    test_data = pd.DataFrame(empl_dict["test_data"])
    test_labels = pd.Series(empl_dict["test_labels"])

    training_data_plot = plot_training_data(
        train_data, train_labels, test_data, test_labels
    )

    # pre-generate all plots for building th core components
    unemployed_activity_start_time_fig = plot_weekly_start_times(
        strava_df, 2014, 2017, work_hours, "Training + Studying"
    )
    employed_activity_start_time_fig = plot_weekly_start_times(
        strava_df, 2018, 2020, work_hours, "Working Full Time"
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

    return (
        unemployed_activity_start_time_fig,
        employed_activity_start_time_fig,
        training_data_plot,
        logreg_train_plot,
        logreg_test_plot,
        lgbm_test_plot,
        lgbm_train_plot,
    )
