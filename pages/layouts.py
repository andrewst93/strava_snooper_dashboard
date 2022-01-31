import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app

STRAVASNOOPER_LOGO = "/assets/images/strava_snooper_wide_logo.png"

# # a contact me button with email, website, github etc.
contact_button = dbc.Button(
    "CONTACT ME",
    id="popover-bottom-target",
    className="mx-2",
    size="lg",
    style={"font-size": "2.2rem", "text-decoration": "none"},
)

page2_button = dbc.Button(
    "PAGE 2",
    href="/pages/page2",
    className="mx-2",
    size="lg",
    style={"font-size": "2.2rem", "text-decoration": "none"},
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
ty_website_link = dbc.NavItem(
    dbc.NavLink(
        "www.ty-andrews.com",
        href="https://ty-andrews.com/",
        target="_blank",
        style={"font-size": "2.0rem", "font-color": "#FFFFFF"},
    )
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
            dbc.Col(html.Img(src=STRAVASNOOPER_LOGO, height="70px"), sm=8, md=6, lg=4),
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
                        page2_button,
                        contact_button,
                        contact_popover,
                        ty_website_link,
                    ],
                    className="ml-auto",
                    navbar=True,  # fill=True, justified=True
                ),
                id="navbar-collapse",
                navbar=True,
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
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
