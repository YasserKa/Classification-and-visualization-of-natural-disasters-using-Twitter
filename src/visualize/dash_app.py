#!/usr/bin/env python3

import ast

import click
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

data_needed = [
    "class",
    "importance",
    "type",
    "display_name",
    "lat",
    "lon",
    "boundingbox",
]
app = Dash(__name__)

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
    # Seperate each data in column
    df[data_needed] = df["locations_info"].to_list()
    df["name"] = df["display_name"].apply(lambda x: x.split(",")[0])
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


def plot(df, app):
    global df_global
    df["count"] = 1
    df_agg = df.groupby(["lon", "lat"], as_index=False).agg(
        {
            "name": "first",
            "count": "sum",
            "id": lambda x: list(x),
            "text": lambda x: list(x),
        }
    )
    df_global = df

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
    fig.update_layout(clickmode="event+select")

    # Group by day
    df["created_at"] = pd.to_datetime(df["created_at"])
    df_agg_day = df.groupby([pd.Grouper(key="created_at", freq="W")])

    dates = df_agg_day.groups.keys()
    import plotly.graph_objects as go

    histo = go.Figure([go.Bar(x=list(dates), y=df_agg_day.count()["count"])])
    histo.update_layout(clickmode="event+select")
    # histo = px.bar(df_agg_day, x=dates, y=df_agg_day.sum())

    app.layout = html.Div(
        className="row",
        children=[
            html.H1(children="Flood Detection"),
            html.Div(children=f"Number of tweets {len(df)}"),
            dcc.Graph(id="geomap", figure=fig),
            dcc.Graph(id="histo", figure=histo),
            dcc.Markdown(children="### Tweets"),
            html.Div(id="selected-data"),
        ],
    )


@app.callback(
    Output("selected-data", "children"),
    # Output("geomap_out", "children"),
    # Output("histo_out", "children"),
    Input("geomap", "selectedData"),
    # Input("histo", "selectedData"),
)
def display_selected_data(selection1):
    if selection1 is None:
        return ""
    indices_selected = [(x["lon"], x["lat"]) for x in selection1["points"]]

    return [
        generate_table(
            df_global[df_global.set_index(["lon", "lat"]).index.isin(indices_selected)][
                ["id", "text"]
            ]
        )
    ]


def generate_table(df, max_rows=10):
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
