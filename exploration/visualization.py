#!/usr/bin/env python3
""" Visualize the output of location extraction model using a geomap and
histogram
"""


import ast
import json
from enum import Enum

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import geopandas as gpd
import pandas as pd
from dash import Dash, Input, Output, html
from dash_extensions.javascript import arrow_function

from flood_detection.data.preprocess import Preprocess


class Region_level(Enum):
    MUNICIPALITIES = "./assets/sweden-municipalities.geojson"
    COUNTIES = "./assets/sweden-counties.geojson"


DEFAULT_REGION_TYPE = Region_level.COUNTIES

selected_region_type = DEFAULT_REGION_TYPE.value
selected_geo = None


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


data_needed = [
    "class",
    "importance",
    "type",
    "display_name",
    "lat",
    "lon",
    "boundingbox",
]

df_global = pd.read_csv(
    "./data/processed_geo/supervisor_annotated_tweets.csv",
    converters={"locations": ast.literal_eval},
)

df_global["locations_info"] = df_global["locations"].apply(get_from_raw_loc)
df_global["loc_smalled_bounding_box"] = df_global["locations_info"].apply(
    get_location_with_lowest_parameter
)
df_global = get_smallest_loc_info(df_global)

preprocess = Preprocess()
df_user_week_uniq = preprocess.get_one_tweet_for_each_user_per_week(
    df_global, per_location=True
)


def get_cluster(df):
    geojson_list = []

    for _, row in df.iterrows():
        geojson_list.append(
            {
                "lat": row["lat"],
                "lon": row["lon"],
                "tooltip": row["loc_name"],
            }
        )

    cluster = dl.GeoJSON(
        data=dlx.dicts_to_geojson(geojson_list),
        cluster=True,
        zoomToBoundsOnClick=True,
        superClusterOptions={"radius": 50},
        id="cluster",
    )
    return cluster


INFO_STYLE = {"margin": "0px"}


def get_hovered_region_info(feature=None):
    if not feature:
        return [html.P("Hover over a region", style=INFO_STYLE)]

    if selected_region_type == Region_level.COUNTIES.value:
        region_name = feature["properties"]["name"]
    else:
        region_name = feature["properties"]["kom_namn"]

    return html.B(f"{region_name}")


def get_selected_region_info(feature=None):
    if not feature:
        return [html.P("Select a region", style=INFO_STYLE)]

    if selected_region_type == Region_level.COUNTIES.value:
        region_name = feature["properties"]["name"]
    else:
        region_name = feature["properties"]["kom_namn"]

    return html.B(f"{region_name}")


def get_intersected_points(feature=None):
    global selected_geo
    if feature is None:
        return df_global

    sweden_geo_df = gpd.GeoDataFrame.from_features([feature])

    geometry = gpd.points_from_xy(df_global["lon"], df_global["lat"])

    gdf_traces = gpd.GeoDataFrame(df_global, geometry=geometry)

    joined_df = gpd.sjoin(
        gdf_traces, sweden_geo_df, how="inner", predicate="intersects"
    )

    return joined_df


with open(selected_region_type, "r") as file:
    sweden_geojson = json.load(file)
    sweden_geo_df = gpd.GeoDataFrame.from_features(sweden_geojson["features"])


def get_choropleth():
    return dl.GeoJSON(
        url=selected_region_type,
        zoomToBounds=True,  # when true, zooms to bounds when data changes (e.g. on load)
        zoomToBoundsOnClick=True,  # when true, zooms to bounds of feature (e.g. polygon) on click
        hoverStyle=arrow_function(dict(weight=5, color="#666", dashArray="")),
        id="choropleth",
    )


cluster = get_cluster(df_global)
choropleth = get_choropleth()
# Create info control.
hover_info = html.Div(
    children=get_hovered_region_info(),
    id="hovered_info",
    className="info",
    style={"position": "absolute", "top": "0px", "right": "10px", "zIndex": "1000"},
)

selected_info = html.Div(
    children=get_selected_region_info(),
    id="selected_info",
    className="info",
    style={"position": "absolute", "top": "30px", "right": "10px", "zIndex": "1000"},
)

radio_region_levels = html.Div(
    [
        dbc.Label("Regions level"),
        dbc.RadioItems(
            options=[
                {"label": "Counties", "value": Region_level.COUNTIES.value},
                {"label": "municipalities", "value": Region_level.MUNICIPALITIES.value},
            ],
            value=selected_region_type,
            id="radio_selected_region",
        ),
    ],
    className="info",
    style={"position": "absolute", "top": "60px", "right": "10px", "zIndex": "1000"},
)

# Create app.
app = Dash(prevent_initial_callbacks=True)
app.layout = html.Div(
    [
        dl.Map(
            children=[
                dl.TileLayer(),
                html.Div([choropleth], id="choropleth_parent"),
                html.Div([cluster], id="cluster_parent"),
                hover_info,
                selected_info,
                radio_region_levels,
            ]
        )
    ],
    style={"width": "100%", "height": "50vh", "margin": "auto", "display": "block"},
    id="map",
)


@app.callback(
    Output("hovered_info", "children"), [Input("choropleth", "hover_feature")]
)
def info_hover_select(feature):
    return get_hovered_region_info(feature)


@app.callback(
    Output("choropleth_parent", "children"),
    [Input("radio_selected_region", "value")],
)
def radio_region_level_update(value):
    global selected_region_type
    selected_region_type = value
    return get_choropleth()


@app.callback(
    Output("cluster_parent", "children"),
    Output("selected_info", "children"),
    [Input("choropleth", "click_feature")],
)
def choropleth_click(feature):
    intersected_points = get_intersected_points(feature)
    return [get_cluster(intersected_points), get_selected_region_info(feature)]


if __name__ == "__main__":
    app.run_server(debug=True)
