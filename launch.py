import sys
import os
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_loading_spinners
import logging

logging.basicConfig()

# src_path = os.path.abspath(os.path.join(".."))
# if src_path not in sys.path:
#     sys.path.append(src_path)

from app import app, server
from src.pages import (
    homepage,
    kudos_prediction,
    employment_prediction,
)

url_bar_and_content_div = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            id="div-loading",
            children=[
                dash_loading_spinners.Pacman(
                    fullscreen=True,
                    id="loading-whole-app",
                    color="#e95420",
                    width=250,
                    speed_multiplier=1.6,
                )
            ],
        ),
        html.Div(id="page-content"),  # , children=homepage.get_layout()),
        dcc.Store(id="strava-data", data=None, storage_type="session"),
        dcc.Store(id="empl-data", data=None, storage_type="session"),
        dcc.Store(id="kudos-actuals", data=None, storage_type="session"),
    ]
)

# "complete" layout from here: https://dash.plotly.com/urls#dynamically-create-a-layout-for-multi-page-app-validation
# app.validation_layout = html.Div(
#     [
#         url_bar_and_content_div,
#         homepage.get_layout(),
#         kudos_prediction.layout,
#         employment_prediction.layout,
#     ]
# )

app.layout = url_bar_and_content_div

# Setup google analytics connection: https://aticoengineering.com/shootin-trouble-in-data-science/google-analytics-in-dash-web-apps/
app.index_string = """<!DOCTYPE html>
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
"""


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def display_page(pathname):
    if pathname == "/":
        return homepage.layout
    if pathname == "/pages/employment-prediction":
        return employment_prediction.layout
    elif pathname == "/pages/kudos-prediction":
        return kudos_prediction.layout
    else:
        return "404"


# Show loading symbol only on first page load
# #idea from here: https://community.plotly.com/t/show-a-full-screen-loading-spinner-on-app-start-up-then-remove-it/60174
@app.callback(
    Output("div-loading", "children"),
    Input("page-content", "loading_state"),
    Input("page-content", "children"),
    [
        State("div-loading", "children"),
    ],
    prevent_initial_call=True,
)
def hide_loading_after_startup(loading_state, page_children, children):

    if children:
        return None
    # raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
