#!/usr/bin/env python3

import ast

import click
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

data_needed = ["class", "importance", "type", "display_name", "lat", "lon"]
app = Dash(__name__)

df_global = pd.DataFrame()


def get_from_raw_loc(row):
    locations = {}
    for name, value in row.items():
        extracted_data = {}
        for data in data_needed:
            if data in value["swedish_loc_info"]:
                extracted_data[data] = value["swedish_loc_info"][data]
            else:
                extracted_data[data] = None
        locations[name] = extracted_data
    return locations


def explode_df(df):
    # Create a row for each location
    df["locations_info"] = df["locations_info"].apply(lambda x: list(x.items()))
    df_exploded = df.explode("locations_info")

    df_exploded = df_exploded[df_exploded["locations_info"].notna()]
    # Seperate each data in column
    df_exploded[["loc_name", "raw_data"]] = df_exploded["locations_info"].to_list()
    df_exploded[["class", "importance", "type", "display_name", "lat", "lon"]] = (
        df_exploded["raw_data"].apply(lambda x: list(x.values())).to_list()
    )
    return df_exploded.astype({"lon": "float", "lat": "float"})


def plot(df, app):
    global df_global
    df["count"] = 1
    df_agg = df.groupby(["lon", "lat"], as_index=False).agg(
        {
            "loc_name": "first",
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
        hover_name=df_agg["loc_name"],
        mapbox_style="carto-positron",
        height=600,
        zoom=3,
        center={"lat": 63.333112, "lon": 16.007205},
    )
    fig.update_layout(clickmode="event+select")

    app.layout = html.Div(
        className="row",
        children=[
            html.H1(children="Flood Detection"),
            html.Div(children=f"Number of tweets {len(df)}"),
            dcc.Graph(id="example-graph", figure=fig),
            dcc.Markdown(children="### Tweets"),
            html.Div(id="selected-data"),
        ],
    )


@app.callback(
    Output("selected-data", "children"), Input("example-graph", "selectedData")
)
def display_selected_data(selectedData):
    if selectedData is None:
        return ""
    indices_selected = [(x["lon"], x["lat"]) for x in selectedData["points"]]

    return generate_table(
        df_global[df_global.set_index(["lon", "lat"]).index.isin(indices_selected)][
            ["id", "text"]
        ]
    )


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
    df_exploded = explode_df(df)

    plot(df_exploded, app)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()