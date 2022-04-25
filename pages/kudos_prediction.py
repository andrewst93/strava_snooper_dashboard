import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import time
import sys
import os

src_path = os.path.abspath(os.path.join(".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from app import app
from pages.layouts import header, footer
from src.visualizations.kudos_prediction_dash_plots import (
    plot_predicted_kudos_value,
    plot_distance_prediction,
    plot_elevation_prediction,
    plot_achievement_prediction,
)
from src.data.kudos_data_load import load_static_kudos_predictions

# initial values
num_followers = 100
distance = 60  # km
custom_name_bool = 0
achievements = 15
elevation = 4  # 100's of meters

static_kudos_predictions = load_static_kudos_predictions("2022-04-19")

kudos_prediction = (
    static_kudos_predictions.loc[
        (static_kudos_predictions.distance == distance)
        & (static_kudos_predictions.achievements == achievements)
        & (static_kudos_predictions.elevation == elevation * 100)
        & (static_kudos_predictions.custom_name_bool == custom_name_bool),
        "kudos_prediction",
    ].iloc[0]
    * num_followers
)


initial_kudos_plot = plot_predicted_kudos_value(round(kudos_prediction))
distance_plot = plot_distance_prediction(
    static_kudos_predictions,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation * 100,
    achievements,
)
elevation_plot = plot_elevation_prediction(
    static_kudos_predictions,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation * 100,
    achievements,
)
achievement_plot = plot_achievement_prediction(
    static_kudos_predictions,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation * 100,
    achievements,
)

kudos_controls = html.Div(
    [
        html.P(
            "Pick what your ride will look like and this will predict how many kudos you'll get."
        ),
        html.P("How Many Followers you have:"),
        dbc.Input(
            type="number", min=5, max=1000, step=1, value=100, id="num-followers"
        ),
        html.Br(),
        dbc.FormGroup(
            [
                dbc.Label("Ride Naming"),
                dbc.RadioItems(
                    options=[
                        {"label": "Standard Ride Name", "value": 0},
                        {"label": "Custom Ride Name", "value": 1},
                    ],
                    value=custom_name_bool,
                    id="custom-name-input",
                    inline=True,
                    style={"text-align": "center"},
                ),
                html.Br(),
                dbc.Label("Ride Distance (km)"),
                dcc.Slider(
                    min=20,
                    max=230,
                    step=10,
                    value=distance,
                    id="distance-slider",
                    marks={int(i): str(i) for i in range(20, 240, 10)},
                ),
                html.Br(),
                dbc.Label("Elevation Gain (hundreds of m)"),
                dcc.Slider(
                    min=1,
                    max=29,
                    step=1,
                    value=elevation,
                    id="elevation-slider",
                    marks={int(i): str(i) for i in range(0, 31, 1)},
                ),
                html.Br(),
                dbc.Label("Strava Achievements on Ride"),
                dcc.Slider(
                    min=0,
                    max=50,
                    step=5,
                    value=achievements,
                    id="achievements-slider",
                    marks={int(i): str(i) for i in range(0, 55, 5)},
                ),
                html.Br(),
            ],
        ),
    ],
)

# DCC card for showing my activitees over time etc.
kudos_prediction_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("How Many Kudos Will You Get on Your Next Ride?")),
        dbc.CardBody(
            children=[
                dbc.Row(
                    [
                        dbc.Col(kudos_controls, width=12, lg=7),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    children=[
                                        dcc.Loading(
                                            children=[
                                                dcc.Graph(
                                                    id="kudos-plot",
                                                    figure=initial_kudos_plot,
                                                    config={"displayModeBar": False},
                                                )
                                            ],
                                            type="circle",
                                            color="#e95420",
                                        ),
                                    ]
                                )
                            ),
                            width=12,
                            lg=5,
                        ),
                    ]
                ),
                dbc.Row(
                    html.H5(
                        "Use these to Maximize your Kudos",
                        style={"textAlign": "center"},
                    ),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.CardGroup(
                                [
                                    dbc.Card(
                                        dbc.CardBody(
                                            children=[
                                                dcc.Loading(
                                                    children=[
                                                        dcc.Graph(
                                                            id="dist-plot",
                                                            figure=distance_plot,
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        )
                                                    ],
                                                    type="circle",
                                                    color="#e95420",
                                                ),
                                            ]
                                        ),
                                    ),
                                    dbc.Card(
                                        dbc.CardBody(
                                            children=[
                                                dcc.Loading(
                                                    children=[
                                                        dcc.Graph(
                                                            id="elev-plot",
                                                            figure=elevation_plot,
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        )
                                                    ],
                                                    type="circle",
                                                    color="#e95420",
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.Card(
                                        dbc.CardBody(
                                            children=[
                                                dcc.Loading(
                                                    children=[
                                                        dcc.Graph(
                                                            id="achiev-plot",
                                                            figure=achievement_plot,
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        )
                                                    ],
                                                    type="circle",
                                                    color="#e95420",
                                                ),
                                            ]
                                        )
                                    ),
                                ]
                            ),
                        ],
                        width=12,
                    )
                ),
            ]
        ),
    ],
    color="#E0E0E0",
)

layout = html.Div(
    [
        header(),
        dbc.Row(
            dbc.Col(
                kudos_prediction_card,
                width={"size": 12, "offset": 0},
                lg={"size": 10, "offset": 1},
            )
        ),
        footer(),
    ]
)


@app.callback(
    Output(component_id="kudos-plot", component_property="figure"),
    Output(component_id="dist-plot", component_property="figure"),
    Output(component_id="elev-plot", component_property="figure"),
    Output(component_id="achiev-plot", component_property="figure"),
    [
        Input(component_id="num-followers", component_property="value"),
        Input(component_id="custom-name-input", component_property="value"),
        Input(component_id="distance-slider", component_property="value"),
        Input(component_id="elevation-slider", component_property="value"),
        Input(component_id="achievements-slider", component_property="value"),
    ],
)
def update_predicted_kudos_number(
    num_followers, custom_name_bool, distance, elevation, achievements
):
    elevation_m = elevation * 100  # get into meters

    perc_followers = static_kudos_predictions.loc[
        (static_kudos_predictions.distance == distance)
        & (static_kudos_predictions.achievements == achievements)
        & (static_kudos_predictions.elevation == elevation_m)
        & (static_kudos_predictions.custom_name_bool == custom_name_bool),
        "kudos_prediction",
    ]

    kudos_prediction = perc_followers.iloc[0] * num_followers

    kudos_value_fig = plot_predicted_kudos_value(round(kudos_prediction))
    distance_value_fig = plot_distance_prediction(
        static_kudos_predictions,
        kudos_prediction,
        num_followers,
        custom_name_bool,
        distance,
        elevation_m,
        achievements,
    )

    elevation_value_fig = plot_elevation_prediction(
        static_kudos_predictions,
        kudos_prediction,
        num_followers,
        custom_name_bool,
        distance,
        elevation_m,
        achievements,
    )

    achievement_value_fig = plot_achievement_prediction(
        static_kudos_predictions,
        kudos_prediction,
        num_followers,
        custom_name_bool,
        distance,
        elevation_m,
        achievements,
    )

    return (
        kudos_value_fig,
        distance_value_fig,
        elevation_value_fig,
        achievement_value_fig,
    )


@app.callback(
    Output("app-2-display-value", "children"), Input("app-2-dropdown", "value")
)
def display_value(value):
    return 'You have selected "{}"'.format(value)
