#!/usr/bin/env python3
""" Visualize the output of location extraction model using a geomap and
histogram
"""


import ast

import click
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from flood_detection.data.preprocess import Preprocess

# OPTIMIZE: Use global state instead of calculating the repatitive values
# (e.g. length of dataframe )

app = Dash(external_stylesheets=[dbc.themes.YETI])

MAX_NUM_META_DATA_LOCATIONS = 5

data_needed = [
    "class",
    "importance",
    "type",
    "display_name",
    "lat",
    "lon",
    "boundingbox",
]

df_global = pd.DataFrame()


def get_from_raw_loc(row):
    locations = {}
    for name, value in row.items():
        if len(value["swedish_loc_info"]) > 0:
            location_info = {
                data: value["swedish_loc_info"][data] for data in data_needed
            }
            locations[name] = location_info
    return locations


def get_smallest_loc_info(df):
    # Create a row for each location
    df["locations_info"] = df["loc_smalled_bounding_box"].apply(
        lambda x: list(x.values())
    )

    df = df[df["locations_info"].str.len() > 0]
    # Separate each data in column
    df.loc[:, data_needed] = df["locations_info"].tolist()

    df.loc[:, "loc_name"] = df["display_name"].apply(lambda x: x.split(",")[0])
    return df.astype({"lon": "float", "lat": "float"})


def get_location_with_lowest_parameter(row):
    lowest_param = 999
    curr_location = {}
    for location in row.values():
        bounding_box = location["boundingbox"]
        param = abs(float(bounding_box[0]) - float(bounding_box[1])) + abs(
            float(bounding_box[2]) - float(bounding_box[3])
        )
        if param < lowest_param:
            lowest_param = param
            curr_location = location

    return curr_location


def get_geomap(df):
    df_group = df_global.groupby(["lon", "lat"], as_index=False, group_keys=True)

    df_agg = df_group.agg(
        {
            "loc_name": "first",
            "count": "sum",
            "selected": "sum",
            "id": lambda x: list(x),
            "processed": lambda x: list(x),
        }
    )

    lisst = []
    # print(df_agg)
    for index, row in df_agg.iterrows():
        lisst.append(
            dl.Minichart(
                lat=row["lat"],
                lon=row["lon"],
                width=10,
                height=10,
                data=[
                    row["count"] - row["selected"],
                    row["selected"],
                ],
                type="pie",
                id=str(index),
                labels=["bl", "df"],
            ),
        )

    fig = dl.Map(
        center=[63.333112, 16.007205],
        zoom=4,
        children=[dl.TileLayer(), *lisst],
        style={
            "width": "100%",
            "height": "50vh",
            "margin": "auto",
            "display": "block",
        },
        id="map",
    )
    return fig


def get_histo(df):
    # Group by day
    df_agg_day = df.groupby(
        [
            pd.Grouper(
                key="created_at",
                freq="W",
            )
        ],
        group_keys=True,
    )

    dates = df_agg_day.groups.keys()
    indices = df_agg_day.groups.values()

    histo = go.Figure(
        [
            go.Bar(
                x=list(dates),
                y=df_agg_day.count()["count"],
                customdata=list(indices),
            )
        ]
    )
    histo.update_layout(clickmode="event+select")
    return histo


def plot(df, app):
    global df_global
    df = df.rename(
        columns={
            "text": "processed",
            "text_raw": "raw",
            "text_translated": "translated",
        }
    )
    df_global = df
    df_global["count"] = 1
    df_global["selected"] = 1

    histo = get_histo(df)
    geomap = get_geomap(df)
    meta_data_html = get_meta_data_html(df)
    CONTENT_STYLE = {
        "margin": "0rem",
    }
    wanted_columns = ["id", "raw", "translated", "processed", "loc_name", "created_at"]
    selected_columns = ["raw", "translated", "processed"]
    options = [{"label": column, "value": column} for column in wanted_columns]
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        # html.H3("Meta Data", className="card-title"),
                                        html.P(
                                            meta_data_html,
                                            id="meta_data",
                                            className="card-text",
                                        ),
                                    ]
                                ),
                                style={"padding": "0px"},
                            ),
                            dbc.Checklist(
                                id="checklist-inline-input",
                                inline=True,
                                options=options,
                                value=selected_columns,
                            ),
                            html.Div(id="tweets", style={"height": "85vh"}),
                        ],
                        style={
                            "width": "50%",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="geomap",
                                children=[geomap],
                                style={"height": "auto"},
                            ),
                            dcc.Graph(id="histo", figure=histo),
                        ],
                        style={
                            "width": "50%",
                        },
                    ),
                ],
                style={"display": "flex"},
            ),
        ],
        style=CONTENT_STYLE,
    )


@app.callback(
    Output("tweets", "children"),
    Output("geomap", "children"),
    Output("meta_data", "children"),
    # Input("geomap", "map"),
    Input("histo", "selectedData"),
    Input("checklist-inline-input", "value"),
)
def display_selected_data(histo_selection, checkbox_checked):
    selected_indices = df_global.index

    for selected_data_fig in [histo_selection]:
        if selected_data_fig and selected_data_fig["points"]:
            selected_indices_fig = sum(
                [p["customdata"] for p in selected_data_fig["points"]], []
            )
            selected_indices = np.intersect1d(selected_indices, selected_indices_fig)

    data_selected = df_global[df_global.index.isin(selected_indices)]
    df_global["selected"] = df_global.index.isin(selected_indices)
    meta_data_html = get_meta_data_html(data_selected)

    return [
        generate_table(data_selected[checkbox_checked]),
        get_geomap(data_selected),
        meta_data_html,
    ]


def get_meta_data_html(data_selected):
    total_data_num = len(df_global)
    total_loc_num = len(df_global["loc_name"].unique())
    created_at_series = pd.to_datetime(data_selected["created_at"]).sort_values()
    oldest_time = created_at_series.iloc[0]
    newest_time = created_at_series.iloc[-1]

    locations_selected_agg = (
        data_selected.groupby(["loc_name"])
        .count()
        .sort_values(by=["count"], ascending=False)
    )
    locations_selected_num = len(locations_selected_agg)

    # Surround number with parenthesis
    def repl(m):
        return f"({m.group(0)})"

    locations_selected_agg["count"] = (
        locations_selected_agg["count"]
        .astype("string")
        .str.replace(r"[0-9]+", repl, regex=True)
    )

    locations_selected = locations_selected_agg.index.str.cat(
        locations_selected_agg["count"], join="left", sep=" "
    ).array

    STYLE = {"margin-top": "0rem", "margin-bottom": "0rem", "float": "left"}
    meta_data = {
        "Tweets": f"Total: {str(total_data_num)}, Selected: {str(len(data_selected))} ",
        " Spans": f"from {str(oldest_time)[:-6]} to {str(newest_time)[:-6]}",
        "Locations": f"Total: {str(total_loc_num)}, Selected: {str(locations_selected_num)} ,",
        "Selected locations": "",
    }
    meta_data_els = []
    for key, value in meta_data.items():
        if key == "Selected locations":
            if locations_selected_num <= MAX_NUM_META_DATA_LOCATIONS:
                meta_data_els.append(html.Div(", ".join(locations_selected) + " | "))
            else:
                meta_data_els.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    ", ".join(
                                        locations_selected[:MAX_NUM_META_DATA_LOCATIONS]
                                    )
                                    + ",",
                                    dbc.Button(
                                        " etc.",
                                        id="other-locations-popover",
                                        className="me-1",
                                        color="link",
                                        style={"padding": "0px"},
                                    ),
                                ],
                                style={**STYLE, "display": "inline"},
                            ),
                            dbc.Popover(
                                dbc.PopoverBody(
                                    ", ".join(
                                        locations_selected[MAX_NUM_META_DATA_LOCATIONS:]
                                    )
                                ),
                                target="other-locations-popover",
                                trigger="click",
                                style={"max-width": "50%"},
                            ),
                        ]
                    )
                )
        else:
            meta_data_els.append(html.Div([html.B(key), ": ", value], style=STYLE))

    return meta_data_els


def generate_table(df, max_rows=20):
    table = [
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody(
            [
                html.Tr([html.Td(str(df.iloc[i][col])) for col in df.columns])
                for i in range(min(len(df), max_rows))
            ]
        ),
    ]
    return dbc.Table(
        table,
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
        style={
            "height": "100%",
            "overflow-y": "scroll",
            "display": "block",
        },
    )


@click.command()
@click.argument("path_to_data", nargs=-1)
def main(path_to_data):
    df = pd.read_csv(
        path_to_data[0],
        converters={"locations": ast.literal_eval},
    )
    df["locations_info"] = df["locations"].apply(get_from_raw_loc)
    df["loc_smalled_bounding_box"] = df["locations_info"].apply(
        get_location_with_lowest_parameter
    )
    df = get_smallest_loc_info(df)

    preprocess = Preprocess()
    df_user_week_uniq = preprocess.get_one_tweet_for_each_user_per_week(
        df, per_location=True
    )

    plot(df_user_week_uniq, app)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
