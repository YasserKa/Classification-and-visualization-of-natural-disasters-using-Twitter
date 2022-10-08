# %%
from src.data.preprocess import remove_not_needed_elements_from_string
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm.notebook import tqdm
from geopy.geocoders import Nominatim, GeoNames
import plotly.express as px
from shapely.geometry import Point
import geopandas as gpd
from urllib.request import urlopen
import json
import ast

# %%
with initialize(version_base=None, config_path="../conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed

df = pd.read_csv("../" + path_to_data)

print(
    f"Number of tweets that explicity mentions Swedish locations\n{len(df[df['mentions_location'] == 1])}/{len(df)}"
)

# %%
tok = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = AutoModel.from_pretrained("KB/bert-base-swedish-cased")
nlp = pipeline(
    "ner",
    model="KB/bert-base-swedish-cased-ner",
    tokenizer="KB/bert-base-swedish-cased-ner",
)


# %%
def get_location_entities(text: str):
    text = remove_not_needed_elements_from_string(text)
    # Remove stopwords
    tokens = nlp(text)
    updated_tokens = []
    for token in tokens:
        if token["word"].startswith("##"):
            try:
                updated_tokens[-1]["word"] += token["word"][2:]
            except Exception:
                continue
        else:
            updated_tokens += [token]
    loc_entities = list(filter(lambda x: x["entity"] == "LOC", updated_tokens))
    return [{entity["word"]: entity["score"]} for entity in loc_entities]


# Use NER on only relevant tweets
df = df[df["relevant"] == 1]
tqdm.pandas(desc="NER NLP")
df["tokens"] = df["raw_text"].progress_apply(get_location_entities)

# %% [markdown]
# The NLP pipeline seems to generate tokens missing it's initial part that's needed for the subsequents that contains "##" at the start

# %% [markdown]
# Now, let's filter out non-swedish locations

# %%
geolocater = Nominatim(user_agent="flood_detection")


# %%
def is_swedish_geo(list_entities):
    for geo in list_entities:
        entity_name = list(geo.keys())[0]
        swedish_location = geolocater.geocode(
            entity_name, country_codes="se", language="en", extratags=True
        )
        if swedish_location is not None:
            #             if swedish_location.raw['importance'] > 0.5:
            # if 'population' in swedish_location.raw['extratags'] and \
            #     int(swedish_location.raw['extratags']['population']) > 1000:
            return swedish_location.raw
    return False


df["has_loc_entities"] = df["tokens"].apply(lambda x: len(x) > 0)
tqdm.pandas(desc="Is Swidish location")
df["loc_ent_is_swedish"] = df["tokens"].progress_apply(is_swedish_geo)
df.to_csv("file1.csv")

# %%
df = pd.read_csv("file1.csv")

confusion_matrix = pd.crosstab(
    df["mentions_location"],
    df["loc_ent_is_swedish"] != "False",
    rownames=["Actual"],
    colnames=["Predicted"],
)
confusion_matrix

# %%
TP = len(df[(df["loc_ent_is_swedish"] != "False") & (df["mentions_location"] == 1)])
TN = len(df[(df["loc_ent_is_swedish"] == "False") & (df["mentions_location"] == 0)])
FP = len(df[(df["loc_ent_is_swedish"] != "False") & (df["mentions_location"] == 0)])
FN = len(df[(df["loc_ent_is_swedish"] == "False") & (df["mentions_location"] == 1)])

print(f"precision {TP/(TP+FP)}")
print(f"recall {TP/(TP+FN)}")
print(f"f1 {TP/(TP+0.5*(FP+FN))}")

# %% [markdown]
# ## Error analysis

# %%
# False negatives
needed_columns = ["id", "tokens", "raw_text", "text"]
false_neg = df[(df["mentions_location"] == 1) & (df["loc_ent_is_swedish"] == "False")][
    needed_columns
]

# %%
# Locations extracted by NER, not recognized by geocoders
false_neg[false_neg["tokens"].apply(lambda x: len(ast.literal_eval(x)) > 0)]


# %%
# Another geocoder
geonames_geolocater = GeoNames("yasser_kaddoura")
print(geonames_geolocater.geocode("Skånska", country="SE", country_bias="SE"))
print(geonames_geolocater.geocode("Kullabygden", country="SE", country_bias="SE"))
print(geonames_geolocater.geocode("Sydkusten", country="SE", country_bias="SE"))

# %%
# Locations not extracted by NER
# Gavla wsn't recognized
# E18 E6 wasn't recognized (Are these places?)
false_neg[false_neg["tokens"].apply(lambda x: len(ast.literal_eval(x)) == 0)]

# %%
# Geocoder seems to identify some of the locations not extracted by NER
# To increase recall, use geocders on text
print(geolocater.geocode("Skånska", country_codes="se", language="en", extratags=True))
print(geolocater.geocode("E6", country_codes="se", language="en", extratags=True))
print(
    geolocater.geocode("Höglandet", country_codes="se", language="en", extratags=True)
)

# %%
# False positives
false_pos = df[(df["mentions_location"] == 0) & (df["loc_ent_is_swedish"] != "False")][
    needed_columns
]
false_pos.head(50)
# Sweden has locations that contains florida and miami
# Idea: Increase precision by:
# - Make more restrictive classification using data extracted from geocoders regarding location (e.g. population, type, popularity)
# - Check non-swedish enteries for that location and if they are (e.g. more popular) filter the swedish entry out


# %%
def parse_raw_text(raw_json):
    if raw_json == "False":
        return pd.Series(5 * [False])
    return pd.Series(
        [
            ast.literal_eval(raw_json)["class"],
            ast.literal_eval(raw_json)["importance"],
            ast.literal_eval(raw_json)["type"],
            ast.literal_eval(raw_json)["display_name"],
            float(ast.literal_eval(raw_json)["lat"]),
            float(ast.literal_eval(raw_json)["lon"]),
        ]
    )


df[
    [
        "loc_class",
        "loc_importance",
        "loc_type",
        "loc_display_name",
        "loc_lat",
        "loc_lon",
    ]
] = df["loc_ent_is_swedish"].apply(parse_raw_text)


# %%
def get_color(row):
    if row["mentions_location"] == 1 and row["loc_ent_is_swedish"] != "False":
        return "blue"
    elif row["mentions_location"] == 0 and row["loc_ent_is_swedish"] != "False":
        return "red"


df["color"] = df.apply(
    get_color,
    axis=1,
)


# %%
# Get the geojson file and load it as a geopandas dataframe
with urlopen(
    "https://raw.githubusercontent.com/ostropunk/swegeojson/master/geodata/kommun/Kommun_RT90_region.json"
) as response:
    sweden_json = json.load(response)

sweden_geo_df = gpd.GeoDataFrame.from_features(sweden_json["features"])
geometry = [Point(x, y) for x, y in zip(df["loc_lon"], df["loc_lat"])]

# %%
fig = px.scatter_mapbox(
    df,
    lat="loc_lat",
    lon="loc_lon",
    hover_name=df["loc_display_name"],
    hover_data=["id"],
    color_discrete_map={"blue": "blue", "red": "red"},
    color="color",
    mapbox_style="carto-positron",
    height=600,
    zoom=3,
    center={"lat": 63.333112, "lon": 16.007205},
)

fig.show()
