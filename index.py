import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# src_path = os.path.abspath(os.path.join(".."))

# if src_path not in sys.path:
#     sys.path.append(src_path)

from app import app, server
from pages import kudos_prediction, layouts, employment_prediction  # homepage,

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

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


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return employment_prediction.layout
    elif pathname == "/pages/kudos-prediction":
        return kudos_prediction.layout
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=True)
