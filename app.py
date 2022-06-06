import dash
import dash_bootstrap_components as dbc

# used for debugging GCP app engine deployment issues
try:
    import googleclouddebugger

    googleclouddebugger.enable(breakpoint_enable_canary=True)
except ImportError:
    pass

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.UNITED],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)  # dbc.themes.UNITED
app.config.suppress_callback_exceptions = True
app.title = "Strava Snooper"
server = app.server

if __name__ == "__main__":
    pass
