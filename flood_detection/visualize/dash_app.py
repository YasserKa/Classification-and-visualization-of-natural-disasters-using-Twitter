#!/usr/bin/env python3
""" Visualize the output of location extraction model using a geomap and
histogram
"""


import ast
import itertools
import math
from enum import Enum

import click
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
from dash_extensions.javascript import arrow_function
from gensim import corpora, models
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from flood_detection.data.preprocess import Language, Preprocess
from flood_detection.predict.text_analysis import (LDA_model,
                                                   get_preprocessed_docs)

# OPTIMIZE: Use global state instead of calculating the repatitive values
# (e.g. length of dataframe )
# WARN: This script is highly unfactorized

app = Dash(external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME],
           prevent_initial_callbacks=True)


class Region_level(Enum):
    MUNICIPALITIES = "./assets/sweden-municipalities.geojson"
    COUNTIES = "./assets/sweden-counties.geojson"


DEFAULT_REGION_TYPE = Region_level.COUNTIES
INFO_STYLE = {"margin": "0px"}

ZOOM_CONFIG = dict(
    center=[63.333112, 16.007205],
    zoom=4,
)

selected_region_type = DEFAULT_REGION_TYPE.value
# Global state to check if histo selection data got changed
global_histo_selection = {}
num_tweets_location = 0
num_tweets_mentioning_sweden = 0
tsne_object = None
tfidf_object = None

# Number of times reset button got clicked
# Plotly only provides the number of times a button got clicked, so a global
# state is needed to check if a button is clicked
current_clicks = 0
clustering_current_clicks = 0

lda_model = None
# Number of topics used on selected tweets
num_lda_topics = 2

# params for DBSCAN clustering in TSNE updates by text input
eps = 1.2
min_samples = 2

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
selected_data = pd.DataFrame()

# Used for sorting table
global_topics_df = pd.DataFrame()
global_tfidf_df = pd.DataFrame()

# When region level changes, this function gets triggered, return
# previous intersected_points
intersected_points = pd.DataFrame()

# When region level changes, this function gets triggered, return
# previous intersected_points
intersected_points = pd.DataFrame()


def get_smallest_loc_info(df):
    # Create a row for each location
    df["locations_info"] = df["loc_smallest_param"].apply(
        lambda x: [x[key] for key in data_needed]
    )

    # Separate each data in column
    df_loc = pd.DataFrame(df["locations_info"].tolist(), columns=data_needed)

    df = pd.concat([df, df_loc], axis=1)

    df.loc[:, "loc_name"] = df["display_name"].apply(lambda x: x.split(",")[0])
    return df.astype({"lon": "float", "lat": "float"})


def get_location_with_lowest_parameter(row):
    global num_tweets_mentioning_sweden
    lowest_param = 999
    curr_location = {}
    for location in row.values():
        location = location["swedish_loc_info"]
        # Not a swedish location
        if len(location) == 0:
            continue
        # This term means county in swedish and doesn't imply a geographic location
        if "Lan" in location["display_name"].split(","):
            continue
        if "Sweden" == location["display_name"].split(",")[0]:
            num_tweets_mentioning_sweden += 1
            continue
        bounding_box = location["boundingbox"]
        param = abs(float(bounding_box[0]) - float(bounding_box[1])) + abs(
            float(bounding_box[2]) - float(bounding_box[3])
        )
        if param < lowest_param:
            lowest_param = param
            curr_location = location

    return curr_location


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
    global df_global
    if feature is None:
        return df_global

    sweden_geo_df = gpd.GeoDataFrame.from_features([feature])

    df = df_global.copy()
    geometry = gpd.points_from_xy(df["lon"], df["lat"])

    gdf_traces = gpd.GeoDataFrame(df, geometry=geometry)

    joined_df = gpd.sjoin(
        gdf_traces, sweden_geo_df, how="inner", predicate="intersects"
    ).rename(columns={"created_at_left": "created_at"})

    return pd.DataFrame(joined_df.drop(columns="geometry"))


def get_choropleth():
    return dl.GeoJSON(
        url=selected_region_type,
        zoomToBounds=False,  # when true, zooms to bounds when data changes (e.g. on load)
        zoomToBoundsOnClick=True,  # when true, zooms to bounds of feature (e.g. polygon) on click
        hoverStyle=arrow_function(dict(weight=5, color="#666", dashArray="")),
        id="choropleth",
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


def get_cluster_points(cluster_point):
    global df_global
    coord = cluster_point["geometry"]["coordinates"]
    if not cluster_point["properties"]["cluster"]:
        return df_global[
            (df_global["lon"] == coord[0]) & (df_global["lat"] == coord[1])
        ]
    cluster_size = cluster_point["properties"]["point_count"]

    df_global["distance"] = (df_global["lon"] - coord[0]) ** 2 + (
        df_global["lat"] - coord[1]
    ) ** 2

    return df_global.sort_values(by=["distance"]).iloc[:cluster_size]


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
                {"label": "counties", "value": Region_level.COUNTIES.value},
                {"label": "municipalities", "value": Region_level.MUNICIPALITIES.value},
            ],
            value=selected_region_type,
            id="radio_selected_region",
        ),
    ],
    className="info",
    style={"position": "absolute", "top": "60px", "right": "10px", "zIndex": "1000"},
)

# resets zoom and selection
reset_button = html.Div(
    [
        dbc.Button(
            "Reset",
            color="light",
            className="me-1",
            id="reset_button",
        ),
    ],
    style={
        "position": "absolute",
        "top": "160px",
        "right": "10px",
        "zIndex": "1000",
    },
)
github_link_button = html.Div(
    [
        dbc.Button(
            html.I(className="fa-brands fa-github"),
            color="light",
            href="https://github.com/YasserKa/Classification-and-visualization-of-natural-disasters-using-Twitter/tree/main",
            external_link=True,
            className="me-0",
            id="github_button",
        ),
    ],
    style={
        "position": "absolute",
        "top": "10px",
        "left": "47px",
        "zIndex": "1000",
        "padding": "2px",
    },
)
text_analysis_button = html.Div(
    [
        dbc.Button(
            [
                dbc.Spinner(
                    html.Div(id="loading-output", children="Text analysis"), size="sm"
                ),
            ],
            color="light",
            className="me-1",
            id="text_analysis_button",
        ),
    ],
    style={
        "padding": "2px",
        "float": "left",
    },
)

num_topics_text_input = html.Div(
    [
        html.P("# topics", style={"margin": "10px", "float": "left"}),
        dbc.Input(
            id="num_topics_input",
            type="number",
            placeholder=f"default: {num_lda_topics}",
            value=num_lda_topics,
            style={"width": "200px"},
        ),
    ]
)


# The output callback is redundent, but it's required by dash
@app.callback(Output("num_topics_input", "value"), [Input("num_topics_input", "value")])
def num_topics_text_input_update(value):
    global num_lda_topics

    # Number of topics should be a positive integer
    if value is None or value < 1:
        num_lda_topics = ""
    else:
        num_lda_topics = math.floor(value)
    return num_lda_topics


eps_text_input = html.Div(
    [
        html.P("eps", style={"margin": "10px", "float": "left"}),
        dbc.Input(
            id="eps_text_input",
            type="number",
            placeholder=f"default: {eps}",
            value=eps,
            style={"width": "75px"},
        ),
    ]
)


# The output callback is redundent, but it's required by dash
@app.callback(Output("eps_text_input", "value"), [Input("eps_text_input", "value")])
def eps_text_input_update(value):
    global eps

    # Number of topics should be a positive integer
    if value is None or value < 1:
        eps = ""
    else:
        eps = value

    return eps


min_samples_text_input = html.Div(
    [
        html.P("min sample", style={"margin": "10px", "float": "left"}),
        dbc.Input(
            id="min_samples_text_input",
            type="number",
            placeholder=f"default: {min_samples}",
            value=min_samples,
            style={"width": "75px"},
        ),
    ]
)


@app.callback(
    Output("min_samples_text_input", "value"),
    [Input("min_samples_text_input", "value")],
)
def min_samples_text_input_update(value):
    global min_samples

    # Number of topics should be a positive integer
    if value is None or value < 1:
        min_samples = ""
    else:
        min_samples = value

    return min_samples


clustering_button = html.Div(
    [
        dbc.Button(
            "DBSCAN",
            color="light",
            className="me-1",
            id="clustering_button",
        ),
    ],
    style={
        "padding": "2px",
        "float": "left",
    },
)


def get_meta_data_html(data_selected):
    global num_tweets_location
    global num_tweets_mentioning_sweden
    total_data_num = len(df_global)
    total_loc_num = len(df_global["loc_name"].unique())
    created_at_series = pd.to_datetime(data_selected["created_at"]).sort_values()
    if len(data_selected) > 0:
        oldest_time = created_at_series.iloc[0]
        newest_time = created_at_series.iloc[-1]
    else:
        oldest_time = ""
        newest_time = ""

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

    if len(data_selected) > 0:
        locations_selected = locations_selected_agg.index.str.cat(
            locations_selected_agg["count"], join="left", sep=" "
        ).array
    else:
        locations_selected = [""]

    STYLE = {"margin-top": "0rem", "margin-bottom": "0rem", "float": "left"}
    meta_data = {
        "Tweets": f"Total: {str(num_tweets_location)}, With location: {str(total_data_num)}, Selected: "
        f"{str(len(data_selected))} ,"
        f"Has word Sweden: {str(num_tweets_mentioning_sweden)} ,",
        "Spans": f"from {str(oldest_time)[:-6]} to {str(newest_time)[:-6]} ,",
        "Locations": f"Total: {str(total_loc_num)}, Selected: {str(locations_selected_num)} ,",
        "Selected locations": "",
    }
    meta_data_els = []
    for key, value in meta_data.items():
        if key == "Selected locations":
            div = [
                ", ".join(
                    locations_selected[
                        : min(len(locations_selected), MAX_NUM_META_DATA_LOCATIONS)
                    ]
                )
            ]
            if locations_selected_num > MAX_NUM_META_DATA_LOCATIONS:
                div += [
                    ",",
                    dbc.Button(
                        " etc.",
                        id="other-locations-popover",
                        className="me-1",
                        color="link",
                        style={"padding": "0px"},
                    ),
                    dbc.Popover(
                        dbc.PopoverBody(
                            ", ".join(locations_selected[MAX_NUM_META_DATA_LOCATIONS:])
                        ),
                        target="other-locations-popover",
                        trigger="click",
                        style={"max-width": "50%"},
                    ),
                ]
            meta_data_els.append(html.Div(div, style={**STYLE, "display": "inline"}))
        else:
            meta_data_els.append(html.Div([html.B(key), ": ", value], style=STYLE))

    return meta_data_els


@app.callback(Output("tweets", "children"), [Input("checklist-inline-input", "value")])
def generate_tweets_table(checkbox_checked):
    global df_global
    PAGE_SIZE = 20

    return dash_table.DataTable(
        df_global[checkbox_checked].iloc[:PAGE_SIZE].to_dict("records"),
        id="datatable-paging",
        columns=[{"name": i, "id": i} for i in sorted(checkbox_checked)],
        page_current=0,
        page_size=PAGE_SIZE,
        page_action="custom",
        style_table={
            "height": "45vh",
            "overflowY": "auto",
        },
        style_cell={
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_cell_conditional=[{"if": {"column_id": "Region"}, "textAlign": "left"}],
        style_data={"whiteSpace": "normal", "height": "auto"},
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#eee",
            }
        ],
        sort_action="custom",
        sort_mode="single",
        sort_by=[],
    )


def generate_tfidf_terms_table(df):

    columns = list(df.columns)
    columns.remove("term_id")
    # place term at the start of the table
    columns.insert(0, columns.pop(columns.index("term")))
    return dash_table.DataTable(
        df[columns].to_dict("records"),
        id="datatable-paging-tfidf",
        columns=[{"name": i, "id": i} for i in columns],
        style_table={
            "height": "35vh",
        },
        style_cell={
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_cell_conditional=[{"if": {"column_id": "Region"}, "textAlign": "left"}],
        style_data={"whiteSpace": "normal", "height": "auto"},
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#eee",
            }
        ],
        sort_action="custom",
        sort_mode="single",
        sort_by=[],
    )


def generate_topics_table(df):

    columns = list(df.columns)
    columns.remove("term_id")
    # place term at the start of the table
    columns.insert(0, columns.pop(columns.index("term")))
    return dash_table.DataTable(
        df[columns].to_dict("records"),
        id="datatable-paging-topics",
        columns=[{"name": i, "id": i} for i in columns],
        style_table={
            "height": "35vh",
            "overflowY": "auto",
        },
        style_cell={
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_cell_conditional=[{"if": {"column_id": "Region"}, "textAlign": "left"}],
        style_data={"whiteSpace": "normal", "height": "auto"},
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#eee",
            }
        ],
        sort_action="custom",
        sort_mode="single",
        sort_by=[],
    )


def get_map(choropleth, cluster):
    return dl.Map(
        **ZOOM_CONFIG,
        children=[
            dl.TileLayer(),
            html.Div([choropleth], id="choropleth_parent"),
            html.Div([cluster], id="cluster_parent"),
            hover_info,
            selected_info,
            radio_region_levels,
            reset_button,
            github_link_button,
        ],
    )


def get_histo(df):
    # Group by day
    created_at = df_global["created_at"].sort_values()
    if len(created_at) > 0:
        time_interval = (created_at.iloc[-1] - created_at.iloc[0]).days
    else:
        time_interval = 0

    selected_df = df_global[df_global.index.isin(df.index)]
    not_selected_df = df_global[~df_global.index.isin(df.index)]

    if time_interval >= 30:
        freq = "M"
    else:
        freq = "D"

    df_agg_day_selected = selected_df.groupby(
        [
            pd.Grouper(
                key="created_at",
                freq=freq,
            )
        ],
        group_keys=True,
    )
    df_agg_day_not_selected = not_selected_df.groupby(
        [
            pd.Grouper(
                key="created_at",
                freq=freq,
            )
        ],
        group_keys=True,
    )

    dates_selected = df_agg_day_selected.groups.keys()
    indices_selected = df_agg_day_selected.groups.values()
    dates_not_selected = df_agg_day_not_selected.groups.keys()
    indices_not_selected = df_agg_day_not_selected.groups.values()

    histo = go.Figure(
        data=[
            go.Bar(
                name="selected",
                x=list(dates_selected),
                y=df_agg_day_selected.count()["count"],
                customdata=list(indices_selected),
            ),
            go.Bar(
                name="not selected",
                x=list(dates_not_selected),
                y=df_agg_day_not_selected.count()["count"],
                customdata=list(indices_not_selected),
            ),
        ],
        layout={
            "margin": go.layout.Margin(
                l=5,  # left margin
                r=5,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            ),
        },
    )
    histo.update_layout(
        clickmode="event+select",
        barmode="stack",
        legend=dict(yanchor="top", orientation="h", y=0.99, xanchor="left", x=0.01),
    )
    return histo


class TSNE_class(object):
    def __init__(self, df):
        # Load the data and create document vectors
        tfidf = TfidfVectorizer()

        self.X = tfidf.fit_transform(df["text"])

        self.X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random", perplexity=5
        ).fit_transform(self.X)
        self.update_clustering()

    def update_clustering(self):
        global eps
        global min_samples
        self.clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(self.X)


class TFIDF_class(object):
    def __init__(self, docs):
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(text) for text in docs]

        self.tfidf_model = models.TfidfModel(corpus)

    def get_top_terms(self, docs) -> pd.DataFrame:
        corpus = [self.dictionary.doc2bow(text) for text in docs]
        corpus_tfidf = self.tfidf_model[corpus]

        # join tweets text into one list
        tfidf_list = list(itertools.chain.from_iterable(list(corpus_tfidf)))

        df_tfidf = (
            pd.DataFrame(tfidf_list, columns=["term_id", "weight"])
            .groupby("term_id", as_index=False)
            .agg(
                weight_mean=pd.NamedAgg(column="weight", aggfunc="mean"),
                weight_count=pd.NamedAgg(column="term_id", aggfunc="count"),
            )
            .sort_values([("weight_count"), ("weight_mean")], ascending=False)
        )
        df_tfidf["term"] = df_tfidf["term_id"].apply(lambda x: self.dictionary[x])

        df_tfidf = df_tfidf.rename(
            columns={
                "weight_count": "count",
                "weight_mean": "mean",
            }
        )
        df_tfidf = df_tfidf.round(5)

        return df_tfidf


def get_scatter(selected_points=[]):
    global tsne_object
    aligned_tweets = []
    # Add breaks to text to create a linewidth
    for tweet in df_global["processed"]:
        index = 0
        lines = []
        while index != -1:
            index = tweet.find(" ", 50)
            lines.append("".join(tweet[:index]))
            tweet = tweet[index + 1 :]
        aligned_tweets.append("<br>".join(lines))
    scatter = go.Figure(
        data=[
            go.Scatter(
                x=tsne_object.X_embedded[:, 0],
                y=tsne_object.X_embedded[:, 1],
                mode="markers",
                hoverinfo="text",
                text=aligned_tweets,
                marker={
                    "color": tsne_object.clusters.labels_,
                    "line_width": 1,
                },
                opacity=0.7,
                selectedpoints=selected_points,
            )
        ],
        layout={
            "margin": go.layout.Margin(
                l=5,  # left margin
                r=5,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            ),
            "clickmode": "event+select",
        },
    )
    return scatter


def plot(df, app):
    global df_global
    global intersected_points
    df = df.rename(
        columns={
            "text": "processed",
            "text_raw": "raw",
            "text_translated": "translated",
        }
    )
    df_global = df

    intersected_points = df

    df_global["count"] = 1
    df_global["selected"] = 1

    df_global["hashtags"] = df_global["raw"].apply(
        lambda text: ", ".join(
            [word for word in str(text).split() if word.startswith("#")]
        )
    )

    histo = get_histo(df)

    cluster = get_cluster(df_global)
    choropleth = get_choropleth()
    map = get_map(choropleth, cluster)
    meta_data_html = get_meta_data_html(df)
    CONTENT_STYLE = {
        "margin": "0rem",
    }
    wanted_columns = [
        "id",
        "raw",
        "translated",
        "processed",
        "hashtags",
        "loc_name",
        "softmax",
        "created_at",
    ]
    selected_columns = ["raw", "translated", "processed"]
    options = [{"label": column, "value": column} for column in wanted_columns]
    scatter = get_scatter(df.index)
    topics_table, tfidf_table, _ = text_anlaysis_button(0)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Checklist(
                                id="checklist-inline-input",
                                inline=True,
                                options=options,
                                value=selected_columns,
                                style={"text-align": "center"},
                            ),
                            html.Div(
                                children=[
                                    generate_tweets_table(selected_columns),
                                ],
                                id="tweets",
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        children=[
                                            html.Div(
                                                [
                                                    clustering_button,
                                                    eps_text_input,
                                                    min_samples_text_input,
                                                ],
                                                style={"display": "inline-flex"},
                                            ),
                                            dcc.Graph(
                                                id="scatter",
                                                figure=scatter,
                                                style={
                                                    "float": "left",
                                                    "width": "100%",
                                                    "height": "90%",
                                                },
                                            ),
                                        ],
                                        style={"width": "50%"},
                                    ),
                                    html.Div(
                                        children=[
                                            html.Div(
                                                [
                                                    text_analysis_button,
                                                    num_topics_text_input,
                                                ],
                                                style={"display": "inline-flex"},
                                            ),
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(
                                                        topics_table,
                                                        id="lda_topics",
                                                        label="LDA Topics",
                                                        label_style={"padding": "5px"},
                                                        style={
                                                            "height": "37vh",
                                                            "overflow-y": "auto",
                                                            "overflow-x": "auto",
                                                        },
                                                    ),
                                                    dbc.Tab(
                                                        tfidf_table,
                                                        id="tf-idf_terms",
                                                        label="TF-IDF terms",
                                                        label_style={"padding": "5px"},
                                                        style={
                                                            "height": "37vh",
                                                            "overflow-y": "auto",
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={
                                            "width": "50%",
                                        },
                                    ),
                                ],
                                style={"display": "inline-flex", "width": "100%"},
                            ),
                        ],
                        style={
                            "width": "50%",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="map",
                                children=[map],
                                style={
                                    "width": "100%",
                                    "height": "50vh",
                                    "margin": "auto",
                                    "display": "block",
                                },
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.P(
                                            meta_data_html,
                                            id="meta_data",
                                            className="card-text",
                                        ),
                                    ],
                                    style={"padding": "0px"},
                                ),
                                style={"padding": "0px"},
                            ),
                            dcc.Graph(
                                id="histo", figure=histo, style={"height": "41%"}
                            ),
                        ],
                        style={
                            "width": "50%",
                        },
                    ),
                    dcc.Store(id="selected_data"),
                ],
                style={"display": "flex"},
            ),
        ],
        style=CONTENT_STYLE,
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
    return [get_choropleth()]


@app.callback(
    Output("lda_topics", "children"),
    Output("tf-idf_terms", "children"),
    Output("loading-output", "children"),
    [
        Input("text_analysis_button", "n_clicks"),
    ],
)
def text_anlaysis_button(text_n_clicks):
    global tfidf_object
    global selected_data
    global num_lda_topics
    global global_topics_df
    global global_tfidf_df

    df = selected_data.rename(columns={"processed": "text"})
    docs = get_preprocessed_docs(df, Language.ENGLISH)

    df_global_ = df_global.rename(columns={"processed": "text"})
    all_docs = get_preprocessed_docs(df_global_, Language.ENGLISH)

    num_topics = 2 if num_lda_topics == "" else num_lda_topics

    lda_model = LDA_model(all_docs, num_topics)
    topics_df = lda_model.get_topics(docs)
    global_topics_df = topics_df

    top_tfidf_df = tfidf_object.get_top_terms(docs)
    global_tfidf_df = top_tfidf_df
    return [
        generate_topics_table(topics_df),
        generate_tfidf_terms_table(top_tfidf_df),
        "Text analysis",
    ]


@app.callback(
    Output("selected_data", "data"),
    Output("selected_info", "children"),
    [
        Input("histo", "selectedData"),
        Input("scatter", "selectedData"),
        Input("choropleth", "click_feature"),
        Input("cluster", "click_feature"),
        Input("reset_button", "n_clicks"),
    ],
)
def display_selected_data(
    histo_selection,
    scatter_selection,
    choropleth_selection,
    cluster_selection,
    n_clicks,
):
    global df_global
    global selected_data
    global current_clicks
    global global_histo_selection
    selected_indices = df_global.index

    if n_clicks is not None and n_clicks > current_clicks:
        selected_data = df_global
        current_clicks = n_clicks
    elif histo_selection not in [None, global_histo_selection]:
        for selected_data_fig in [histo_selection]:
            if selected_data_fig and selected_data_fig["points"]:
                selected_indices_fig = sum(
                    [p["customdata"] for p in selected_data_fig["points"]], []
                )
                selected_indices = np.intersect1d(
                    selected_indices, selected_indices_fig
                )
                selected_data = df_global[df_global.index.isin(selected_indices)]
        global_histo_selection = histo_selection
    elif scatter_selection is not None:
        if len(scatter_selection["points"]) > 0:
            selected_indices = [
                point["pointIndex"] for point in scatter_selection["points"]
            ]
            selected_data = df_global[df_global.index.isin(selected_indices)]
        # Selection on scatter triggers this function with no selected points,
        # making this conditional needed
        else:
            pass
    elif cluster_selection is not None:
        selected_data = get_cluster_points(cluster_selection)
    elif choropleth_selection is not None:
        selected_data = get_intersected_points(choropleth_selection)
    else:
        selected_data = df_global

    df_global["selected"] = df_global.index.isin(selected_data)

    return [
        selected_data.to_json(date_format="iso", orient="split"),
        get_selected_region_info(choropleth_selection),
    ]


@app.callback(
    Output("cluster_parent", "children"),
    Output("histo", "figure"),
    Output("scatter", "figure"),
    Output("meta_data", "children"),
    [
        Input("selected_data", "data"),
        Input("clustering_button", "n_clicks"),
    ],
)
def update_map(selected_data, n_clicks):
    global df_global
    global clustering_current_clicks

    if n_clicks is not None and n_clicks > clustering_current_clicks:
        tsne_object.update_clustering()
        clustering_current_clicks = n_clicks

    if selected_data is not None:
        selected_data = pd.read_json(selected_data, orient="split")
    else:
        selected_data = df_global
    selected_points = selected_data.index

    return [
        get_cluster(selected_data),
        get_histo(selected_data),
        get_scatter(selected_points),
        get_meta_data_html(selected_data),
    ]


@app.callback(
    Output("datatable-paging", "data"),
    [
        Input("selected_data", "data"),
        Input("datatable-paging", "page_current"),
        Input("datatable-paging", "page_size"),
        Input("datatable-paging", "sort_by"),
        Input("checklist-inline-input", "value"),
    ],
)
def update_table(selected_data, page_current, page_size, sort_by, columns_selected):
    global df_global
    if selected_data is not None:
        selected_data = pd.read_json(selected_data, orient="split")
    else:
        selected_data = df_global

    # Need to round it again, otherwise the decimal numbers contain 0000000001
    selected_data = selected_data.round(3)

    if len(sort_by):
        selected_data = selected_data.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )

    return (
        selected_data[columns_selected]
        .iloc[page_current * page_size : (page_current + 1) * page_size]
        .to_dict("records")
    )


@app.callback(
    Output("datatable-paging-topics", "data"),
    [
        Input("datatable-paging-topics", "sort_by"),
    ],
)
def update_topics_table(sort_by):
    global global_topics_df
    # Need to round it again, otherwise the decimal numbers contain 0000000001

    if len(sort_by):
        df = global_topics_df.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )
    else:
        df = global_topics_df

    return df.to_dict("records")


@app.callback(
    Output("datatable-paging-tfidf", "data"),
    [
        Input("datatable-paging-tfidf", "sort_by"),
    ],
)
def update_tfidf_table(sort_by):
    global global_tfidf_df
    # Need to round it again, otherwise the decimal numbers contain 0000000001

    if len(sort_by):
        df = global_tfidf_df.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )
    else:
        df = global_tfidf_df

    return df.to_dict("records")


@click.command()
@click.argument("path_to_data", nargs=-1)
def main(path_to_data):
    global num_tweets_location
    global selected_data
    global tsne_object
    global lda_model
    global num_lda_topics
    global tfidf_object

    print("Loading and processing data")
    df = pd.read_csv(
        path_to_data[0],
        converters={
            "swedish_locations": ast.literal_eval,
        },
    )
    df["loc_smallest_param"] = df["swedish_locations"].apply(
        get_location_with_lowest_parameter
    )
    num_tweets_location = len(df)

    df = df[df["loc_smallest_param"].str.len() > 0].reset_index()

    df = get_smallest_loc_info(df)

    preprocess = Preprocess()
    df_user_week_uniq = preprocess.get_one_tweet_for_each_user_per_week(
        df, per_location=True
    )
    df_user_week_uniq = df_user_week_uniq.round(3)
    selected_data = df_user_week_uniq

    print("Doing text analysis")

    docs = get_preprocessed_docs(df_user_week_uniq, Language.ENGLISH)

    tsne_object = TSNE_class(df_user_week_uniq)
    df = df_user_week_uniq.rename(columns={"processed": "text"})
    docs = get_preprocessed_docs(df, Language.ENGLISH)
    tfidf_object = TFIDF_class(docs)

    print("Plotting")
    plot(df_user_week_uniq, app)
    app.run_server(host="0.0.0.0")


if __name__ == "__main__":
    main()
