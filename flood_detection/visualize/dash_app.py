#!/usr/bin/env python3
""" Visualize the output of location extraction model using a geomap and
histogram
"""


import ast

import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__)

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

    df.loc[:, "name"] = df["display_name"].apply(lambda x: x.split(",")[0])
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
            "name": "first",
            "count": "sum",
            "id": lambda x: list(x),
            "text": lambda x: list(x),
        }
    )
    indecies = df_group.groups.values()
    fig = px.scatter_mapbox(
        df_agg,
        lat="lat",
        lon="lon",
        size="count",
        hover_name=df_agg["name"],
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
    df["created_at"] = pd.to_datetime(df["created_at"])
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
    df_global = df

    geomap = get_geomap(df)
    histo = get_histo(df)

    app.layout = html.Div(
        className="row",
        children=[
            html.H1(children="Flood Detection"),
            html.Div(children=f"Total number of tweets: {len(df)}"),
            html.Div(id="tweets_num_picked"),
            dcc.Graph(id="geomap", figure=geomap),
            dcc.Graph(id="histo", figure=histo),
            dcc.Markdown(children="### Tweets Selected"),
            html.Div(id="tweets"),
        ],
    )


@app.callback(
    Output("tweets", "children"),
    Output("geomap", "figure"),
    Output("tweets_num_picked", "children"),
    Input("geomap", "selectedData"),
    Input("histo", "selectedData"),
)
def display_selected_data(geomap_selection, histo_selection):
    selected_indices = df_global.index

    for selected_data_fig in [geomap_selection, histo_selection]:
        if selected_data_fig and selected_data_fig["points"]:
            selected_indices_fig = sum(
                [p["customdata"] for p in selected_data_fig["points"]], []
            )
            selected_indices = np.intersect1d(selected_indices, selected_indices_fig)

    data = df_global[df_global.index.isin(selected_indices)]
    tweets_num_picked = f"Number of tweets picked: {len(data)}"
    return [
        generate_table(data[["id", "text"]]),
        get_geomap(data),
        tweets_num_picked,
    ]


def generate_table(df, max_rows=20):
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in df.columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(str(df.iloc[i][col])) for col in df.columns])
                    for i in range(min(len(df), max_rows))
                ]
            ),
        ]
    )


@click.command()
@click.argument("path_to_data", nargs=-1)
def main(path_to_data):
    df = pd.read_csv(path_to_data[0], converters={"locations": ast.literal_eval})
    df["locations_info"] = df["locations"].apply(get_from_raw_loc)
    df["loc_smalled_bounding_box"] = df["locations_info"].apply(
        get_location_with_lowest_parameter
    )
    df = get_smallest_loc_info(df)

    plot(df, app)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
