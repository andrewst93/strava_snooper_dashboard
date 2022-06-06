from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc

from app import app

STRAVASNOOPER_LOGO = "/assets/images/strava_snooper_wide_logo.png"


def blank_placeholder_plot(background_color="white"):
    """Creates an empty plot to be used on page load before callbacks complete,
    allows page load and all figure updates/data loading to be managed in callbacks
    without empty plots being shown.

    Args:
        background_color (str, optional): What color the background should match to be "invisible" until updated.
                                         Defaults to "white".

    Returns:
        px.Scatter : Empty plot with lines removed and formatting updated to be blank.
    """

    blank_plot = px.scatter(x=[0, 1], y=[0, 1])
    blank_plot.update_xaxes(showgrid=False, showticklabels=False, visible=False)
    blank_plot.update_yaxes(showgrid=False, showticklabels=False, visible=False)

    blank_plot.update_traces(marker=dict(color=background_color))

    blank_plot.layout.plot_bgcolor = background_color
    blank_plot.layout.paper_bgcolor = background_color
    return blank_plot


# # a contact me button with email, website, github etc.
contact_button = dbc.Button(
    "CONTACT ME",
    id="popover-bottom-target",
    className="me-1",
    size="lg",
    # style={
    #     "font-size": "1.5rem",
    #     # "padding": "50px10px",
    # },  # "padding - first val top/bot, 2nd left/right"
)

page2_button = dbc.Button(
    "PAGE 2",
    href="/pages/page2",
    className="mx-2",
    size="lg",
    style={"font-size": "1.5rem", "text-decoration": "none"},
)  #

# a contact me button with email, website, github etc.
contact_popover = dbc.Popover(
    [
        dbc.PopoverHeader(
            dbc.Row("Get in Touch", justify="center", style={"font-size": "2.5rem"})
        ),
        dbc.PopoverBody(
            [
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            html.P(
                                "ty.elgin.andrews@gmail.com",
                                style={"font-size": "1.25rem"},
                            ),
                            size="md",
                        ),
                        dbc.Button(
                            "LinkedIn",
                            href="https://www.linkedin.com/in/ty-andrews-237256a0/",
                            size="md",
                            target="_blank",
                            style={"font-size": "2.0rem"},
                        ),
                        dbc.Button(
                            "Github",
                            href="https://github.com/andrewst93",
                            size="md",
                            target="_blank",
                            style={"font-size": "2.0rem"},
                        ),
                    ],
                    vertical=True,
                    # color='secondary'
                )
            ]
        ),
    ],
    id=f"popover-bottom",
    target=f"popover-bottom-target",
    placement="bottom-end",
)

# make a reuseable navitem for the different examples
ty_website_link = dbc.NavLink(
    "www.ty-andrews.com",
    href="https://ty-andrews.com/",
    external_link=True,
    style={
        # "font-size": "1.2rem",
        "font-color": "#FFFFFF",
    },
)


# custom navbar based on https://getbootstrap.com/docs/4.1/examples/dashboard/
def footer():
    return dbc.Navbar(
        [
            dbc.Col(html.Img(src=STRAVASNOOPER_LOGO, height="50px"), width=4),
            dbc.Col(
                "Copyright " + "\u00A9" + " 2021 Ty Andrews. All Rights Reserved",
                style={"textAlign": "center", "font-size": "1.5rem", "color": "white"},
                sm=2,
                md=4,
            ),
        ],
        color="secondary",
        dark=True,
    )


# custom navbar based on https://getbootstrap.com/docs/4.1/examples/dashboard/
def header():

    return dbc.Navbar(
        [
            dbc.Col(
                html.Img(src=STRAVASNOOPER_LOGO, style={"width": "100%"}),
                xs=12,
                sm=8,
                md=6,
                lg=4,
            ),
            dbc.Col(
                "What does Strava know about you?",
                style={"textAlign": "center", "font-size": "2rem", "color": "white"},
                sm=0,
                md=2,
                lg=4,
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.NavLink(
                                "Home",
                                active="exact",
                                href="/",
                                style={"text-decoration": "none"},
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                "Employment Prediction",
                                active="exact",
                                href="/pages/employment-prediction",
                                style={"text-decoration": "none", "font-weight": 700},
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                "Predict Your Kudos",
                                active="exact",
                                href="/pages/kudos-prediction",
                                style={"text-decoration": "none", "font-weight": 700},
                            )
                        ),
                        # dbc.NavItem(ty_website_link),
                        dbc.NavItem(contact_button),
                        contact_popover,
                    ],
                    className="ml-auto",
                    navbar=True,
                    pills=True,
                ),
                id="navbar-collapse",
                navbar=True,
                className="g-0",
            ),
        ],
        color="primary",
        dark=True,
        sticky="top",
        expand="lg",
    )


@app.callback(
    Output(f"popover-bottom", "is_open"),
    [Input(f"popover-bottom-target", "n_clicks")],
    [State(f"popover-bottom", "is_open")],
)  # (toggle_popover)
def toggle_popover(n, is_open):
    """Allows for Contact Me pop over to be open/closed on page.

    Args:
        n (int): how many times the pop has been clicked
        is_open (bool): Whether the popover is open or closed

    Returns:
        bool: Changes the is_open state to the opposite to open/close the pop over.
    """
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """Manages collapsible navbar on small screens.

    Args:
        n (int): Number of clicks on the menu
        is_open (bool): State of whether the menu is open or closed

    Returns:
        bool: updates the state of the navbar to open/close it
    """
    if n:
        return not is_open
    return is_open
