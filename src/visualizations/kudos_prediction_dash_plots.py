import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score

from src.data.kudos_data_load import load_static_kudos_predictions

ORANGE = "#FFA400"
BLUE = "#009FFD"


def plot_predicted_kudos_value(kudos):

    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=kudos,
            title={"text": "Predicted <br>Kudos"},
            number={"font": {"size": 135}},
            # number={"suffix": " kudos"},
            # delta = {'position': "top", 'reference': 320},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    return fig


def plot_distance_prediction(
    kudos_pred_df,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation,
    achievements,
):

    dist_plot = px.line(
        x=kudos_pred_df.loc[
            (kudos_pred_df.achievements == achievements)
            & (kudos_pred_df.elevation == elevation)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "distance",
        ],
        y=kudos_pred_df.loc[
            (kudos_pred_df.achievements == achievements)
            & (kudos_pred_df.elevation == elevation)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "kudos_prediction",
        ]
        * num_followers,
        labels=dict(x="<b>Distance</b> (km)", y="Predicted Kudos"),
        color_discrete_sequence=[ORANGE],
    )

    dist_plot.add_trace(
        go.Scatter(
            x=[distance],
            y=[kudos_prediction],
            mode="markers",
            name="Prediction",
            marker=dict(color=BLUE, size=14, line=dict(width=0)),
        )
    )

    max_perc_kudos = kudos_pred_df.loc[
        (kudos_pred_df.elevation == elevation)
        & (kudos_pred_df.achievements == achievements)
        & (kudos_pred_df.custom_name_bool == custom_name_bool),
        "kudos_prediction",
    ].max()
    min_val = kudos_pred_df.loc[
        (kudos_pred_df.elevation == elevation)
        & (kudos_pred_df.achievements == achievements)
        & (kudos_pred_df.custom_name_bool == custom_name_bool)
        & (kudos_pred_df.kudos_prediction == max_perc_kudos),
        "distance",
    ].min()

    dist_plot.add_trace(
        go.Scatter(
            x=[min_val],
            y=[max_perc_kudos * num_followers],
            mode="markers",
            name="Max Kudos",
            marker=dict(color="lime", size=8),
        )
    )

    dist_plot.update_layout(
        title=f"<b>Kudos</b> vs. <b>Distance</b>",
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1.0,
        ),
    )

    return dist_plot


def plot_elevation_prediction(
    kudos_pred_df,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation,
    achievements,
):

    elevation_plot = px.line(
        x=kudos_pred_df.loc[
            (kudos_pred_df.achievements == achievements)
            & (kudos_pred_df.distance == distance)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "elevation",
        ],
        y=kudos_pred_df.loc[
            (kudos_pred_df.achievements == achievements)
            & (kudos_pred_df.distance == distance)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "kudos_prediction",
        ]
        * num_followers,
        labels=dict(x="<b>Elevation Gain</b> (m)", y="Predicted Kudos"),
        color_discrete_sequence=[ORANGE],
    )

    elevation_plot.add_trace(
        go.Scatter(
            x=[elevation],
            y=[kudos_prediction],
            mode="markers",
            name="Prediction",
            marker=dict(color=BLUE, size=14, line=dict(width=0)),
        )
    )

    max_perc_kudos = kudos_pred_df.loc[
        (kudos_pred_df.achievements == achievements)
        & (kudos_pred_df.distance == distance)
        & (kudos_pred_df.custom_name_bool == custom_name_bool),
        "kudos_prediction",
    ].max()
    min_val = kudos_pred_df.loc[
        (kudos_pred_df.achievements == achievements)
        & (kudos_pred_df.distance == distance)
        & (kudos_pred_df.custom_name_bool == custom_name_bool)
        & (kudos_pred_df.kudos_prediction == max_perc_kudos),
        "elevation",
    ].min()

    elevation_plot.add_trace(
        go.Scatter(
            x=[min_val],
            y=[max_perc_kudos * num_followers],
            mode="markers",
            name="Max Kudos",
            marker=dict(color="lime", size=8),
        )
    )

    elevation_plot.update_layout(
        title=f"<b>Kudos</b> vs. <b>Elevation</b>",
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1.0,
        ),
    )

    return elevation_plot


def plot_achievement_prediction(
    kudos_pred_df,
    kudos_prediction,
    num_followers,
    custom_name_bool,
    distance,
    elevation,
    achievements,
):

    achievement_plot = px.line(
        x=kudos_pred_df.loc[
            (kudos_pred_df.elevation == elevation)
            & (kudos_pred_df.distance == distance)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "achievements",
        ],
        y=kudos_pred_df.loc[
            (kudos_pred_df.elevation == elevation)
            & (kudos_pred_df.distance == distance)
            & (kudos_pred_df.custom_name_bool == custom_name_bool),
            "kudos_prediction",
        ]
        * num_followers,
        labels=dict(x="<b>Achievements</b>", y="Predicted Kudos"),
        color_discrete_sequence=[ORANGE],
    )

    achievement_plot.add_trace(
        go.Scatter(
            x=[achievements],
            y=[kudos_prediction],
            mode="markers",
            name="Prediction",
            marker=dict(color=BLUE, size=14, line=dict(width=0)),
        )
    )

    max_perc_kudos = kudos_pred_df.loc[
        (kudos_pred_df.elevation == elevation)
        & (kudos_pred_df.distance == distance)
        & (kudos_pred_df.custom_name_bool == custom_name_bool),
        "kudos_prediction",
    ].max()
    min_val = kudos_pred_df.loc[
        (kudos_pred_df.elevation == elevation)
        & (kudos_pred_df.distance == distance)
        & (kudos_pred_df.custom_name_bool == custom_name_bool)
        & (kudos_pred_df.kudos_prediction == max_perc_kudos),
        "achievements",
    ].min()

    achievement_plot.add_trace(
        go.Scatter(
            x=[min_val],
            y=[max_perc_kudos * num_followers],
            mode="markers",
            name="Max Kudos",
            marker=dict(color="lime", size=8),
        )
    )

    achievement_plot.update_layout(
        title=f"<b>Kudos</b> vs. <b>Achievements</b>",
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1.0,
        ),
    )

    return achievement_plot
