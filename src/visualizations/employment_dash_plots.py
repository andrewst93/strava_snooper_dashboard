import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score

from src.data.strava_data_load_preprocess import (
    load_week_start_times_data,
    load_lgbm_model_results,
    load_logreg_model_results,
    load_lgbm_heatmap,
    load_logreg_heatmap,
    load_lgbm_model,
    load_logreg_model,
)


ACTIVITY_COLOR_MAP = {
    "AlpineSki": "#636EFA",
    "Ride": "#00CC96",
    "NordicSki": "#EF553B",
    "Run": "#AB63FA",
    "Swim": "#FFA15A",
    "Ski": "#19D3F3",
    "Workout": "#FF6692",
    "WeightTraining": "#B6E880",
}


def activity_pie_plot(df, val, label, title):
    """Builds a plotly pie plot broken by activity.

    Args:
        df (DataFrame): raw data to include in the pie plot
        val (string): which column to plot.
        label (string): what the lables should display in the legend,
        title (string): Title to be displayed on the plot

    Returns:
        [Plotly Express Figure]: The complete pie plot of activity types
    """

    fig = px.pie(
        df,
        values=val,
        names=label,
        # color_discrete_sequence= kaha_color_list,
        title=title,
        labels={"type": "Activity Type"},
    )
    fig.update_traces(textposition="inside", textinfo="label+percent", sort=False)
    fig.update_layout(
        showlegend=True,
        title_x=0.5,
        title_font_size=25,
        font_size=15,
        legend={"traceorder": "normal"},
    )

    return fig


def plot_lgbm_model_predictions(
    train_data, plot_data, labels, data_set, lgbm_model=None
):
    """Generates the LGBM results plot using plotly express and prediction region mask.

    Returns:
        Plotly Express Figure: the LGBM model results plot including prediction region mask.
    """
    try:
        lgbm_results = load_lgbm_model_results(data_set)
    except FileNotFoundError:
        print("No LGBM " + data_set + " Results Found, generating")
        lgbm_model = load_lgbm_model()
        prediction = lgbm_model.predict(
            train_data, num_iteration=lgbm_model.best_iteration
        )
        lgbm_results = pd.DataFrame({"label": labels, "prediction": prediction})

    acc = accuracy_score(lgbm_results.label, lgbm_results.prediction.round())

    # results = pd.DataFrame({'label': labels, 'prediction': prediction}, columns = ['label','prediction']).to_csv(os.path.join('data', 'lgbm_model_' + f'{acc*100:.1f}_acc'))

    print(f"Accuracy: {acc:.3f}")

    x_min, x_max = plot_data["morn"].min() - 0.02, plot_data["morn"].max() + 0.02
    y_min, y_max = plot_data["aft"].min() - 0.02, plot_data["aft"].max() + 0.02

    data_plot = px.scatter(
        x=plot_data["morn"],
        y=plot_data["aft"],
        color=np.where(lgbm_results.prediction.round() == 1, "unemployed", "employed"),
        color_discrete_map={
            "unemployed": "#009FFD",
            "employed": "#FFA400",
        },
        symbol=np.where(
            lgbm_results.prediction.round() == np.array(lgbm_results.label),
            "Correct",
            "Incorrect",
        ),
        symbol_map={"Correct": "circle", "Incorrect": "x"},
        labels={"color": "Label", "symbol": "Model Prediction"},
    ).update_traces(marker=dict(size=10, line_width=2, line_color="black"))
    data_plot.update_layout(
        # title=f'Employment Predictions from Weekday Activities Started During Work Hours<br>Model Accuracy: {acc*100:.0f}%',
        title=f"LightGBM Model Accuracy: {acc*100:.0f}%",
        xaxis=dict(
            title="Activities During Morning Work Hours",
            tickformat=",.0%",
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title="Activities During Afternoon Work Hours",
            tickformat=",.0%",
            range=[y_min, y_max],
        ),
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1,
        ),
    )

    # ensure both train and plot data are matched to prevent breaking of the plotly figure.
    if len(train_data.iloc[0]) == len(plot_data.iloc[0]):

        try:
            lgbm_heatmap = load_lgbm_heatmap(data_set)

        except IndexError:
            # no exisiting heatmap, create classification confidence heat map
            print("No Heatmap values for " + data_set + " found, generating...")
            h = 0.005
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            y_ = np.arange(y_min, y_max, h)

            Z = lgbm_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # save off heatmap values
            heat_values = []
            for x in range(0, len(xx[0])):
                for y in range(0, len(y_)):
                    heat_values.append([xx[0][x], y_[y], Z[y][x]])

            lgbm_heatmap = pd.DataFrame(heat_values, columns=["X", "Y", "Z"])
            lgbm_heatmap.to_csv(
                os.path.join("data", "processed", data_set + "_lgbm_model_heatmap.csv")
            )

        data_plot.add_trace(
            go.Heatmap(
                x=lgbm_heatmap.X,
                y=lgbm_heatmap.Y,
                z=lgbm_heatmap.Z,
                colorscale=[
                    [0, "#FFA400"],
                    [0.4, "#FFA400"],
                    [0.6, "#009FFD"],
                    [1, "#009FFD"],
                ],  # [[0,'#FFA400'],  [1,'#009FFD']], #[0.45, '#009FFD'], [0.55,'#FFA400'],
                opacity=0.3,
                showscale=False,
            )
        )

    return data_plot


def plot_logreg_model_predictions(data, labels, data_set):
    """Generate the plotly express plot of the logisitc regression model results including prediction
    confidence mask values.

    Args:
        data (DataFrame): dataset including months and working/not working info
        labels (list): classification labels to be displayed in the plots legend
        data_set (string): which dataset to pull for display.

    Returns:
        Plotly Express Figure: The final formatted plot ready for display.
    """

    # first try to load already processed data, if not use the models to generate
    try:
        logreg_results = load_logreg_model_results(data_set)
    except FileNotFoundError:
        print("No logreg " + data_set + " Results Found, generating")
        logreg_model = load_logreg_model()
        prediction = logreg_model.predict(data)
        logreg_results = pd.DataFrame({"label": labels, "prediction": prediction})

    acc = accuracy_score(logreg_results.label, logreg_results.prediction)

    x_min, x_max = data["morn"].min() - 0.02, data["morn"].max() + 0.02
    y_min, y_max = data["aft"].min() - 0.02, data["aft"].max() + 0.02

    print(f"Accuracy: {acc:.3f}")

    # results = pd.DataFrame({'label': labels, 'prediction': prediction}, columns = ['label','prediction']).to_csv(os.path.join('data', 'logreg_model_' + f'{acc*100:.1f}_acc'))

    data_plot = px.scatter(
        x=data["morn"],
        y=data["aft"],
        color=np.where(logreg_results.label == 1, "unemployed", "employed"),
        color_discrete_map={
            "unemployed": "#009FFD",
            "employed": "#FFA400",
        },
        symbol=np.where(
            logreg_results.prediction == np.array(logreg_results.label),
            "Correct",
            "Incorrect",
        ),
        symbol_map={"Correct": "circle", "Incorrect": "x"},
        labels={"color": "Label", "symbol": "Model Prediction"},
    ).update_traces(marker=dict(size=10, line_width=2, line_color="black"))
    data_plot.update_layout(
        # title=f'Logistic Regression Model<br>Model Accuracy: {acc*100:.0f}%',
        title=f"Logistic Regression Model Accuracy: {acc*100:.0f}%",
        xaxis=dict(
            title="Activities During Morning Work Hours",
            tickformat=",.0%",
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title="Activities During Afternoon Work Hours",
            tickformat=",.0%",
            range=[y_min, y_max],
        ),
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1,
        ),
    )

    # first try to load already processed data, if not use the models to generate
    try:
        logreg_heatmap = load_logreg_heatmap(data_set)
    except IndexError:
        # no exisiting heatmap Create classification confidence heat map
        logreg_model = load_logreg_model()
        print("No Heatmap values for " + data_set + " found, generating...")
        h = 0.005
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        y_ = np.arange(y_min, y_max, h)

        Z = logreg_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        Z = Z.reshape(xx.shape)
        # save off heatmap values
        heat_values = []
        for x in range(0, len(xx[0])):
            for y in range(0, len(y_)):
                heat_values.append([xx[0][x], y_[y], Z[y][x]])

        logreg_heatmap = pd.DataFrame(heat_values, columns=["X", "Y", "Z"])
        logreg_heatmap.to_csv(
            os.path.join("data", "processed", data_set + "_logreg_model_heatmap.csv")
        )

    data_plot.add_trace(
        go.Heatmap(
            x=logreg_heatmap.X,
            y=logreg_heatmap.Y,
            z=logreg_heatmap.Z,
            colorscale=[
                [0, "#009FFD"],
                [0.45, "#009FFD"],
                [0.55, "#FFA400"],
                [1, "#FFA400"],
            ],  # [[0,'#FFA400'],  [1,'#009FFD']], #[0.45, '#009FFD'], [0.55,'#FFA400'],
            opacity=0.3,
            showscale=False,
        )
    )

    return data_plot


def plot_training_data(train_data, train_labels, test_data, test_labels):

    combined_data = pd.concat([train_data, test_data])
    combined_labels = pd.concat([train_labels, test_labels])

    x_min, x_max = (
        combined_data["morn"].min() - 0.02,
        combined_data["morn"].max() + 0.02,
    )
    y_min, y_max = combined_data["aft"].min() - 0.02, combined_data["aft"].max() + 0.02

    data_plot = px.scatter(
        x=combined_data["morn"],
        y=combined_data["aft"],
        color=combined_labels,
        color_discrete_map={"unemployed": "#009FFD", "employed": "#FFA400"},
        labels={
            "color": "Label",
        },
    ).update_traces(marker=dict(size=10, line_width=2, line_color="black"))

    data_plot.update_layout(
        title=f"Data Set of Employed & <br>Unemployed Months",
        xaxis=dict(
            title="Activities During Morning Work Hours",
            tickformat=",.0%",
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title="Activities During Afternoon Work Hours",
            tickformat=",.0%",
            range=[y_min, y_max],
        ),
        title_x=0.5,
        title_font_size=17,
        font_size=12,
        margin=dict(l=5, r=5, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1,
        ),
    )

    return data_plot


def plot_weekly_start_times(activity_df, start_year, end_year, work_hours, title_descr):

    start_time_fig = go.Figure()

    work_perc = 0

    # Load or generate the data for activity start times across an entire year
    try:
        summary_week_start_times = load_week_start_times_data()
    except FileNotFoundError:
        print("No Yearly Week Summary Data Found, generating it now...")
        summary_week_start_times = {}
        for year in range(
            pd.DatetimeIndex(activity_df.start_date_local).year.min(),
            pd.DatetimeIndex(activity_df.start_date_local).year.max() + 1,
        ):
            weekly_start_times = generate_weekly_start_time_dict(activity_df, year)
            summary_week_start_times[str(year)] = weekly_start_times
        json.dump(
            summary_week_start_times,
            open(
                os.path.join(
                    os.getcwd(), "data", "processed", "yearly_week_start_times.json"
                ),
                "w",
            ),
        )

    for year in range(start_year, end_year + 1):

        week_days = np.array(24 * [0])
        weekend_days = np.array(24 * [0])

        # Calculate what eprcentag of workout start times occur at each hour in the day.
        for day in range(0, 5):
            week_days += summary_week_start_times[str(year)][str(day)]
        week_days_perc = week_days / sum(week_days)
        week_days_perc = pd.DataFrame(data=week_days_perc, columns=["exercise_start"])

        for time_span in work_hours:
            work_perc += sum(
                week_days_perc.exercise_start.iloc[time_span[0] - 1 : time_span[1]]
            )
            start_time_fig.add_vrect(
                x0=time_span[0],
                x1=time_span[1] + 1,
                line_width=0,
                fillcolor="orange",
                opacity=0.05,
            )

        # process and get percentage of activities on weekend days over each hour
        for day in range(5, 7):
            weekend_days += summary_week_start_times[str(year)][str(day)]

        weekend_days_perc = weekend_days / sum(weekend_days)
        weekend_days_perc = pd.DataFrame(
            data=weekend_days_perc, columns=["exercise_start"]
        )

        start_time_fig.add_trace(
            go.Scatter(
                x=week_days_perc.index,
                y=week_days_perc["exercise_start"] * 100,
                name=f"{year}",
                mode="lines",
                line_color="#FFA400",
            )
        )
        # start_time_fig.add_trace(go.Scatter(
        #                                 x=weekend_days_perc.index,
        #                                 y=weekend_days_perc['exercise_start']*100,
        #                                 name=f'{year} Weekends',
        #                                 mode='lines',
        #                                 line_color='#009FFD'
        #                                 ))

    avg_work_percent = work_perc / (end_year - start_year + 1)

    start_time_fig.update_layout(
        title=f"{start_year} - {end_year}: {title_descr}<br>{avg_work_percent*100:.0f}% of Weekday Activities Started During Work Hours",
        xaxis=dict(title="Activity Start Time (24hr time)"),
        yaxis=dict(title="Percentage of Activities", ticksuffix="%"),
        title_x=0.5,
        title_font_size=15,
        font_size=10,
        # legend_x=0.85, legend_y=1,
        legend_title_text="Weekdays",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1,
        ),
    )

    return start_time_fig


def plot_eda_data(input_data, year_range, y_label, group_by):

    data = input_data[
        (pd.to_datetime(input_data.start_time) > (str(year_range[0]) + "-1-1"))
        & (pd.to_datetime(input_data.start_time) < (str(year_range[1]) + "-1-1"))
    ]  # and (pd.to_datetime(input_data.start_time) < pd.to_datetime(str(year_range[1]) + '-1-1'))]

    group_map = {
        "Day": "D1",
        "Week": 604800000,  # plotly doesn't have week so this is number of seconds in week
        "Month": "M1",
        "Quarter": "M3",
        "Year": "M12",
    }

    fig = px.histogram(
        data,
        x="start_time",
        y=y_label,
        histfunc="sum",
        color="type",
        color_discrete_map=ACTIVITY_COLOR_MAP,
        labels={
            "distance_raw_km": "Distance (km)",
            "type": "Activity Type",
            "start_time": "Date",
            "elapsed_time_raw_hrs": "Elapsed Time (hrs)",
            "moving_time_raw_hrs": "Moving Time (hrs)",
            "elevation_gain": "Elevation Gain (m)",
            "calories": "Calories",
        },
    )
    fig.update_traces(xbins_size=group_map[group_by])
    fig.update_layout(
        bargap=0.1,
        title="Interactive Activity Data",
        showlegend=True,
        title_x=0.5,
        title_font_size=17,
        font_size=12,
        legend={"traceorder": "normal"},
    )
    fig.update_xaxes(
        showgrid=True,
        ticklabelmode="period",
        dtick="M4",
        tickformat="%b\n%Y",
        # range=[datetime(year_range[0],1,1), datetime(year_range[1],1,1)],
    )
    fig.update_yaxes(autorange=True)

    return fig


def plot_activity_histogram(data):

    # fig = go.Histogram(x=data['start_time'],
    #                 xbins=dict(
    #                 start='2013-1-1',
    #                 end='2022-1-1',
    #                 size='M1'), # M18 stands for 18 months
    #                 autobinx=False
    #                 )
    fig = px.histogram(
        data, x="start_time", y="distance_raw_km", histfunc="sum", color="type"
    )
    fig.update_traces(xbins_size="M1")
    fig.update_xaxes(
        showgrid=True, ticklabelmode="period", dtick="M4", tickformat="%b\n%Y"
    )
    fig.update_layout(bargap=0.1, barmode="stack")

    return fig
