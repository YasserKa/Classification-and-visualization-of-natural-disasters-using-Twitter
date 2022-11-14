#!/usr/bin/env python3
""" Visualize the output of location extraction model using a geomap and
histogram
"""


import ast

import click
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from flood_detection.data.preprocess import Preprocess

app = Dash(external_stylesheets=[dbc.themes.YETI])

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
    df["count"] = 1
    df_group = df.groupby(["lon", "lat"], as_index=False, group_keys=True)

    df_agg = df_group.agg(
        {
            "loc_name": "first",
            "count": "sum",
            "id": lambda x: list(x),
            "processed": lambda x: list(x),
        }
    )
    indecies = df_group.groups.values()
    fig = px.scatter_mapbox(
        df_agg,
        lat="lat",
        lon="lon",
        size="count",
        hover_name=df_agg["loc_name"],
        mapbox_style="carto-positron",
        height=600,
        zoom=3,
        center={"lat": 63.333112, "lon": 16.007205},
    )
    fig.update_traces(customdata=list(indecies))
    fig.update_layout(clickmode="event+select")
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

    geomap = get_geomap(df)
    histo = get_histo(df)
    meta_data_html = get_meta_data_html(df)
    CONTENT_STYLE = {
        "margin-top": "2rem",
        "margin-left": "2rem",
        "margin-right": "2rem",
    }
    wanted_columns = ["id", "raw", "translated", "processed", "loc_name", "created_at"]
    selected_columns = ["id", "raw", "translated", "processed"]
    options = [{"label": column, "value": column} for column in wanted_columns]
    app.layout = html.Div(
        [
            html.H1("Flood Detection"),
            html.Hr(),
            html.P(id="standalone-radio-check-output"),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H3("Meta Data", className="card-title"),
                                html.P(
                                    meta_data_html,
                                    id="meta_data",
                                    className="card-text",
                                ),
                            ]
                        ),
                        style={
                            "width": "40rem",
                            "height": "40rem",
                            "margin-right": "auto",
                            "margin-left": "auto",
                        },
                    ),
                    dcc.Graph(
                        id="geomap",
                        figure=geomap,
                        style={"width": "50rem", "height": "40rem"},
                    ),
                ],
                style={"display": "flex"},
            ),
            dcc.Graph(id="histo", figure=histo),
            dcc.Markdown(children="### Tweets Selected"),
            dbc.Checklist(
                id="checklist-inline-input",
                inline=True,
                options=options,
                value=selected_columns,
            ),
            html.Div(id="tweets"),
        ],
        style=CONTENT_STYLE,
    )


@app.callback(
    Output("tweets", "children"),
    Output("geomap", "figure"),
    Output("meta_data", "children"),
    Input("geomap", "selectedData"),
    Input("histo", "selectedData"),
    Input("checklist-inline-input", "value"),
)
def display_selected_data(geomap_selection, histo_selection, checkbox_checked):
    selected_indices = df_global.index

    for selected_data_fig in [geomap_selection, histo_selection]:
        if selected_data_fig and selected_data_fig["points"]:
            selected_indices_fig = sum(
                [p["customdata"] for p in selected_data_fig["points"]], []
            )
            selected_indices = np.intersect1d(selected_indices, selected_indices_fig)

    data_selected = df_global[df_global.index.isin(selected_indices)]
    meta_data_html = get_meta_data_html(data_selected)

    return [
        generate_table(data_selected[checkbox_checked]),
        get_geomap(data_selected),
        meta_data_html,
    ]


def get_meta_data_html(data_selected):
    total_data_num = len(df_global)
    created_at_series = pd.to_datetime(data_selected["created_at"]).sort_values()
    oldest_time = created_at_series.iloc[0]
    newest_time = created_at_series.iloc[-1]
    locations_selected = data_selected["loc_name"].unique()
    locations_selected_num = len(locations_selected)

    if locations_selected_num > 60:
        selected_locations_str = ", ".join(locations_selected[:60]) + ", ..."
    else:
        selected_locations_str = ", ".join(locations_selected)

    STYLE = {"margin-top": "0rem", "margin-bottom": "0rem"}
    meta_data = {
        "Total tweets": str(total_data_num),
        "Selected tweets": str(len(data_selected)),
        "Number of selected locations": str(len(locations_selected)),
        "Selected locations": selected_locations_str,
        "Tweets span from": str(oldest_time)[:-6],
        "Tweets span to": str(newest_time)[:-6],
    }
    return [
        html.P([html.B(key), ": ", value], style=STYLE)
        for key, value in meta_data.items()
    ]


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
    return dbc.Table(table, striped=True, bordered=True, hover=True)


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
