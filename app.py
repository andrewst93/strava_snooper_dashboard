import dash
import dash_bootstrap_components as dbc

# used for debugging GCP app engine deployment issues
try:
    import googleclouddebugger

    googleclouddebugger.enable(breakpoint_enable_canary=True)
except ImportError:
    pass

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])  # dbc.themes.UNITED
app.title = "Strava Snooper"
server = app.server
app.config.suppress_callback_exceptions = True
