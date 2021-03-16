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
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

from sklearn import datasets
from sklearn.cluster import KMeans

from strava_data_load_preprocess import load_strava_activity_data, generate_employment_prediction_model_data,\
                                         generate_weekly_start_time_dict, load_employment_model_data, load_lgbm_model,\
                                         load_logreg_model, load_week_start_times_data, preprocess_employment_model_data
from dash_plots import activity_pie_plot, plot_lgbm_model_predictions, plot_logreg_model_predictions, \
                        plot_weekly_start_times, plot_eda_data, plot_activity_histogram, plot_training_data

try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=True
  )
except ImportError:
  pass

# ----------- CONSTANTS -----------------------------------#

#What's the definition of "work hours", activiteis are grouped down 9-11 is any activities started 9:00 - 11:59
work_hours = [[9,11],[13,16]]

#------------ END CONSTANTS -----------------------------#

iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

# Load sample data
data_file_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
print('Loading Strava Data: ' + data_file_path)
start = time.process_time()
raw_files_dict = load_strava_activity_data(data_file_path)

print(f'\tTook {time.process_time()- start:.2f}s')

num_activities = len(raw_files_dict['TyAndrews'].type)

# print('Generating Pie Plot and Scatter.')
# start = time.process_time()
# activity_type_pie_plot = activity_pie_plot(
#     df=raw_files_dict['TyAndrews'],
#     val="act_type_perc_time",
#     label='type',
#     title=f'Activity Distribution: {num_activities} Total Activities'
# )

activity_over_time = plot_eda_data(raw_files_dict['TyAndrews'], [2013,2021], 'distance', 'Month')
# activity_histogram = plot_activity_histogram(raw_files_dict['TyAndrews'])

css_file = r"assets\dash_bootstrap_united.css"

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.UNITED]) #dbc.themes.UNITED
app.title = 'Strava Snooper'
server = app.server
app.config.suppress_callback_exceptions = True

STRAVASNOOPER_LOGO = "/assets/images/strava_snooper_wide_logo.png"
# make a reuseable navitem for the different examples
ty_website_link = dbc.NavItem(dbc.NavLink("www.ty-andrews.com", href="https://ty-andrews.com/", target="_blank", style={'font-size':'2.0rem', 'font-color':'#FFFFFF'}))

# # make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Entry 1"),
        dbc.DropdownMenuItem("Entry 2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Entry 3"),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
    style={'font-size': '1.5rem'}
)

contact_popover = dbc.Popover(
        [
            dbc.PopoverHeader(dbc.Row("Get in Touch", justify='center', style={'font-size': '2.5rem'})),
            dbc.PopoverBody(
                [dbc.ButtonGroup(
                    [dbc.Button(html.P('ty.elgin.andrews@gmail.com', style={'font-size': '1.25rem'}), size='md'),
                    dbc.Button('LinkedIn', href='https://www.linkedin.com/in/ty-andrews-237256a0/', size='md', target='_blank', style={'font-size': '2.0rem'}),
                    dbc.Button('Github', href='https://github.com/andrewst93', size='md', target='_blank', style={'font-size': '2.0rem'})],
                    vertical=True,
                    # color='secondary'
                )]
            ),
        ],
        id=f"popover-bottom",
        target=f"popover-bottom-target",
        placement='bottom-end',
    )
contact_button = dbc.Button(
        f"CONTACT ME",
        id=f"popover-bottom-target",
        className="mx-2",
        size = 'lg'
    )


# custom navbar based on https://getbootstrap.com/docs/4.1/examples/dashboard/
header = dbc.Navbar(
    [
        dbc.Col(html.Img(src=STRAVASNOOPER_LOGO, height="70px"), sm=8,md=6, lg=4),
        # dbc.Col("Strava Snooper", style={'font-size':'4rem','color':'white'}, sm=2, md=3, align="start"), 
        dbc.Col("What does Strava know about you?", style={'textAlign': 'center', 'font-size': '2rem', 'color':'white'}, sm=0, md=2, lg=4),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav(
                [
                ty_website_link, 
                contact_button, contact_popover], className="ml-auto", navbar=True, #fill=True, justified=True
            ),
            id="navbar-collapse",
            navbar=True,
        )
    ],
    color="primary",
    dark=True,
    sticky='top',
    expand='lg'
)

jumbotron = dbc.Jumbotron(
    [
        html.H6("Millions of people upload their activities to Strava every day. This got me wondering...", className="display-6"),
        html.Hr(className="my-2"),
        html.H5(
            "What does Strava know about us from our uploaded activities?", className="display-5"
        ),
        html.P("For a full description of the analysis below see the full write up here."),
        html.P(dbc.Button("LEARN MORE", size='lg', href="https://ty-andrews.com/post/2021-02-23-what-does-strava-know-about-me/", color="primary", target="_blank")), #, className="lead"
    ],
    className='dash-bootstrap'
)

activity_controls = html.Div(
    [
        html.P("All of my past 8 years of activities uploaded to Strava were exported. "
                "From 2014-2017 I was training and going to school. 2017 to now I have been working "
                "full time."),
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="eda-variable",
                    options=[
                        # {"label": col, "value": col} for col in raw_files_dict['TyAndrews'].columns
                        {"label": col[0], "value": col[1]} for col in [['Activity Distance', 'distance_raw_km'], ['Elapsed Time', 'elapsed_time_raw_hrs'], ['Moving Time','moving_time_raw_hrs'], ['Elevation Gain','elevation_gain']]
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
                        {"label": col, "value": col} for col in ['Day', 'Week', 'Month', 'Quarter', 'Year']
                    ],
                    value="Month",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Year Range"),
                dcc.RangeSlider(id="year-selector", min=2013, max=2022, step=1, value=[2013,2022], 
                    marks={2013:'2013',
                    2014:'\'14',
                    2015: '\'15',
                    2016:'\'16',
                    2017:'\'17',
                    2018:'\'18',
                    2019:'\'19',
                    2020:'\'20',
                    2021:'\'21', 
                    2022:'2022'
                    }
                ),
            ]
        ),
    ],
    # body=True,
)

data_intro_card = dbc.Card(
    [dbc.CardHeader(
                html.H4("My Strava Data")
    ),
    dbc.CardBody(
        dbc.Row(
            [dbc.Col(activity_controls, width=3),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='eda-plot', figure=activity_over_time))), width= 9)]
        )
    )],
    color='#E0E0E0',
)

unemployed_activity_start_time_fig = plot_weekly_start_times(raw_files_dict['TyAndrews'], 2014, 2017, work_hours, 'Training + Studying')
employed_activity_start_time_fig = plot_weekly_start_times(raw_files_dict['TyAndrews'], 2018, 2020, work_hours, 'Working Full Time')

train_employment_data, test_employment_data = load_employment_model_data()

train_data, train_labels = preprocess_employment_model_data(train_employment_data, work_hours)
test_data, test_labels = preprocess_employment_model_data(test_employment_data, work_hours)

training_data_plot = plot_training_data(train_data, train_labels, test_data, test_labels)

lgbm_train_plot = plot_lgbm_model_predictions(train_data[['morn','aft']], train_data[['morn','aft']], train_labels, 'train') #optional: lgbm_model, 
lgbm_test_plot = plot_lgbm_model_predictions(test_data[['morn','aft']], test_data[['morn','aft']], test_labels, 'test')
logreg_train_plot = plot_logreg_model_predictions(train_data[['morn','aft']], train_labels, 'train') #Optional: logreg_model, 
logreg_test_plot = plot_logreg_model_predictions(test_data[['morn','aft']], test_labels, 'test')

employment_hypoth = dbc.Card(
    [
        dbc.CardHeader(html.H4("Does Strava know whether I'm employed?")),
        dbc.CardBody(
            [dbc.Row(
                [dbc.Col(
                    [html.P("I wondered if Strava was able to determine which of their users are employed based on their workout habits. "
                            "This could be useful to know for a number of reasons such as assessing potential subscribers or improving targeted marketing."),
                    html.Hr(),
                    html.H6("Hypothesis:"),
                    html.P('If a users mid-week activities commonly start outside of work hours (9-12AM, 1-4PM) they are working.'),
                    html.H6("First Look at the Data:"),
                    html.P('From knowing what years I worked I can see that there is a significant trend that '
                            ' when I\'m working fewer activities started during work hours.')
                    ], width=3
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody(
                        [dbc.Row(html.H5("First Look at the Data"), justify='center'),
                        dbc.Row(
                            [dbc.Col(dcc.Graph(id='unempl-explore-plot', figure=unemployed_activity_start_time_fig), width = 6),
                            dbc.Col(dcc.Graph(id='empl-explore-plot', figure=employed_activity_start_time_fig), width = 6)]
                        )]
                    )),
                    width=9
                )]
            ), 
            html.Hr(className="my-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [html.H4("Training Data"),
                        html.P("To apply a machine learning approach to this problem I tackled the data cleaning and prep. "
                                "I decided to split the data into monthly data and create two features, % of weekday activities started during morning work hours, "
                                " and percent started during afternoon hours. For more info on this decision see my blog post on it here:"),
                        html.P(dbc.Button("BLOG POST", size='lg', href="https://ty-andrews.com/post/2021-02-23-what-does-strava-know-about-me/", color="primary", target="_blank")),
                        ], width=3
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id='train-data-plot', figure=training_data_plot), width = 12),
                                        # dbc.Col(dcc.Graph(id='empl-explore-plot', figure=employed_activity_start_time_fig), width = 6)])
                                        ]
                                    )
                                ]
                            )
                        ),
                        width=9
                    )
                ]
            ),
            html.Hr(className="my-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [html.H4("Logistic Regression & LightGBM Models"),
                        html.P("Taking into account the small data size a logistic regression model and Light Gradient Boosted Machine (LGBM) "
                                "were trained on a 70/30 train test split. After some tuning the LGBM showed better accuracy but with more over fitting."),
                        html.Hr(),
                        html.H4('Conclusion'),
                        html.P("These results are promising that with such a simple approach achieved results significantly better than a randomized model."),
                        html.Hr(),
                        html.H4("Future Improvements"),
                        html.P('- as expected it\'s clear increasing the dataset size would help with reducing overfitting'),
                        html.P('- outlier detection to remove months that don\'t have enough activities'),
                        html.P('- including activities started in evenings, lunch times and mornings as features')
                        ], width=3
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    # dbc.Row(html.H5("Logistic Regression Model"), justify='center'),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [dbc.Row(html.H5("Training Data - 70%"), justify='center'),
                                                dcc.Graph(id='logreg-train-data-plot', figure=logreg_train_plot, style={'height': '350px'})], 
                                                width = 6),
                                            dbc.Col(
                                                [dbc.Row(html.H5("Test Data - 30%"), justify='center'),
                                                dcc.Graph(id='logreg-test-data-plot', figure=logreg_test_plot, style={'height': '350px'})],
                                                width = 6)
                                        ], justify='center'
                                    ),
                                    # dbc.Row(html.H5("LightGBM Model"), justify='center'),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id='lgbm-train-data-plot', figure=lgbm_train_plot, style={'height': '350px'}), width = 6),
                                            dbc.Col(dcc.Graph(id='lgbm-test-data-plot', figure=lgbm_test_plot, style={'height': '350px'}), width = 6)
                                        ],
                                    )
                                ]
                            )
                        ),
                        width=9
                    )
                ]
            ),
        ]),
    ],
    color='#E0E0E0',
)

# custom navbar based on https://getbootstrap.com/docs/4.1/examples/dashboard/
footer = dbc.Navbar(
    [
        # dbc.Col(html.Img(src=STRAVASNOOPER_LOGO, height="70px"), width=4),
        # dbc.Col("Strava Snooper", style={'font-size':'4rem','color':'white'}, sm=2, md=3, align="start"), 
        dbc.Col(html.Img(src=STRAVASNOOPER_LOGO, height="50px"), width=4),
        dbc.Col("Copyright " + u"\u00A9" + " 2021 Ty Andrews. All Rights Reserved", style={'textAlign': 'center', 'font-size': '1.5rem', 'color':'white'}, sm=2, md=4), 
        # dbc.NavbarToggler(id="navbar-toggler2"),
        #     dbc.Collapse(
        #         dbc.Nav(
        #             [dbc.Button('Contact Me', size='lg', color='primary', href="ty.elgin.andrews@gmail.com", target="_blank")], className="ml-auto", navbar=True, 
        #         ),
        #         id="navbar-collapse2",
        #         navbar=True,
        #     ),
    ],
    color="secondary",
    dark=True,
)

# Setp google analytics connection: https://aticoengineering.com/shootin-trouble-in-data-science/google-analytics-in-dash-web-apps/
app.index_string = '''<!DOCTYPE html>
<html>
<head>

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9X0F2S7RJG"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-9X0F2S7RJG');
</script>

{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

app.layout = html.Div(
    [   
        header,
        jumbotron,
        dbc.Row(dbc.Col(data_intro_card, width={"size": 10, "offset": 1})),
        dbc.Row(dbc.Col(employment_hypoth, width={"size": 10, "offset": 1})),
        footer
    ],
)

@app.callback(
        Output(f"popover-bottom", "is_open"),
        [Input(f"popover-bottom-target", "n_clicks")],
        [State(f"popover-bottom", "is_open")],
)#(toggle_popover)

def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("eda-plot", "figure"),
    [
        Input("eda-variable", "value"),
        Input("year-selector", "value"),
        Input('eda-group', 'value')
    ],
)
def update_eda_plot(y_label, year_range, group_by):
    return plot_eda_data(raw_files_dict['TyAndrews'], year_range, y_label, group_by)

@app.callback(
    Output("update-plot", "figure"),
    [
        # Input("x-variable", "value"),
        Input("y-variable", "value"),
        Input("year-selector", "value"),
    ],
)
def make_graph(x, y, n_clusters):
    # minimal input validation, make sure there's at least one cluster
    km = KMeans(n_clusters=max(n_clusters, 1))
    df = iris.loc[:, [x, y]]
    km.fit(df.values)
    df["cluster"] = km.labels_

    centers = km.cluster_centers_

    data = [
        go.Scatter(
            x=df.loc[df.cluster == c, x],
            y=df.loc[df.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

    return go.Figure(data=data, layout=layout)


# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in iris.columns
    ]


# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
    filter_options
)
app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
    filter_options
)

if __name__ == "__main__":
    app.run_server(debug=True) #, port=8051
