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
import ast

# %%
with initialize(version_base=None, config_path="../conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed

df = pd.read_csv("../" + path_to_data)
# Use NER on only relevant tweets
df = df[df["relevant"] == 1]

print(
    f"Number of tweets that explicity mentions Swedish locations\n{len(df[df['mentions_location'] == 1])}/{len(df)}"
)

transformer = "KB/bert-base-swedish-cased-ner"
transformer = "KBLab/bert-base-swedish-cased-ner"

# %%
tok = AutoTokenizer.from_pretrained(transformer)
model = AutoModel.from_pretrained(transformer)
nlp = pipeline(
    "ner",
    model=transformer,
    tokenizer=transformer,
    # Default ignore list is ["O"] which are tokens needed to extract locations
    ignore_labels=[],
)

geolocater: Nominatim = Nominatim(user_agent="flood_detection")
# geonames_geolocater = GeoNames("yasser_kaddoura")


# %%
def get_location_entities(text: str):
    text = remove_not_needed_elements_from_string(text, remove_numbers=False)
    tokens = nlp(text)
    merged_tokens = []
    for token in tokens:
        if token["word"].startswith("##"):
            merged_tokens[-1]["word"] += token["word"][2:]
        else:
            merged_tokens += [token]
    # Remove not needed tokens
    return list(filter(lambda x: x["entity"] != "O", merged_tokens))


tqdm.pandas(desc="NER NLP")
df["tokens"] = df["raw_text"].progress_apply(get_location_entities)


# %%
def get_location_tokens(tokens: list[dict]) -> dict[str, dict[str, float]]:
    loc_tokens: dict[str, dict[str, float]] = {}
    for token in tokens:
        if token["entity"] == "LOC":
            loc_tokens[token["word"]] = {"ner_score": token["score"]}
    return loc_tokens


df["locations"] = df["tokens"].apply(get_location_tokens)


# %%
def is_swedish_geo(locations: dict[str, dict]) -> dict[str, dict]:
    swedish_locations = locations.copy()
    for name in locations:
        swedish_location = geolocater.geocode(
            name, country_codes="se", language="en", extratags=True  # pyright: ignore
        )

        # swedish_location = geonames_geolocater.geocode(name, country="SE", country_bias="SE")

        if swedish_location is not None:
            swedish_locations[name]["swedish_loc_info"] = swedish_location.raw
        else:
            swedish_locations[name]["swedish_loc_info"] = {}

    return swedish_locations


tqdm.pandas(desc="Is Swidish location")
df["locations"] = df["locations"].progress_apply(is_swedish_geo)
df.to_csv("file1.csv")

# %%
# Convert locations to json
df = pd.read_csv("file1.csv", converters={"locations": ast.literal_eval})


# %%
def has_swedish_loc(location_row) -> bool:
    for loc in location_row:
        if len(location_row[loc]["swedish_loc_info"]) > 0:
            return True
    return False


TP = df[(df["locations"].apply(has_swedish_loc)) & (df["mentions_location"] == 1)]
TN = df[(~df["locations"].apply(has_swedish_loc)) & (df["mentions_location"] == 0)]
FP = df[(df["locations"].apply(has_swedish_loc)) & (df["mentions_location"] == 0)]
FN = df[(~df["locations"].apply(has_swedish_loc)) & (df["mentions_location"] == 1)]

confusion_matrix = pd.crosstab(
    df["mentions_location"],
    df["locations"].apply(has_swedish_loc),
    rownames=["Actual"],
    colnames=["Predicted"],
)
print(confusion_matrix)

print(f"precision {len(TP)/(len(TP)+len(FP))}")
print(f"recall {len(TP)/(len(TP)+len(FN))}")
print(f"f1 {len(TP)/(len(TP)+0.5*(len(FP)+len(FN)))}")

# %% [markdown]
# ## Error analysis

# %%
# False negatives
needed_columns = ["id", "locations", "raw_text", "text"]

# Locations extracted by NER, not recognized by geocoders
FN[FN["locations"].apply(lambda x: len(x) > 0)][needed_columns]

# %%
# Locations not extracted by NER
FN[FN["locations"].apply(lambda x: len(x) == 0)][needed_columns]

# %%
# Geocoder seems to identify some of the locations not extracted by NER
# To increase recall, use geocders on text directly
print(
    geolocater.geocode("Åker–Nyhom", country_codes="se", language="en", extratags=True)
)
print(
    geolocater.geocode("Höglandet", country_codes="se", language="en", extratags=True)
)


# %%
# False positives
def func(row):
    return [
        key if len(value["swedish_loc_info"]) > 0 else None
        for key, value in row.items()
    ]


x = FP.copy()
x["loc_names"] = FP["locations"].apply(func)
x[["loc_names", "id"]]
# Sweden has locations that contains florida and miami
# Idea: Increase precision by:
# - Make more restrictive classification using data extracted from geocoders regarding location (e.g. population, type, popularity)
# - Check non-swedish enteries for that location and if they are (e.g. more popular) filter the swedish entry out


# %%
x = geolocater.geocode("Vita", country_codes="se", language="en", extratags=True).raw
print(x)

# %%
data_needed = ["class", "importance", "type", "display_name", "lat", "lon"]

pd.set_option("max_colwidth", 800)


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


df["locations_info"] = df["locations"].apply(get_from_raw_loc)

# %%
# Create a row for each location
df["locations_info"] = df["locations_info"].apply(lambda x: list(x.items()))
df_exploded = df.explode("locations_info")

df_exploded = df_exploded[df_exploded["locations_info"].notna()]
# Seperate each data in column
df_exploded[["loc_name", "raw_data"]] = df_exploded["locations_info"].to_list()
df_exploded[["class", "importance", "type", "display_name", "lat", "lon"]] = (
    df_exploded["raw_data"].apply(lambda x: list(x.values())).to_list()
)
df_exploded = df_exploded.astype({"lon": "float", "lat": "float"})


def get_color(row):
    if row["mentions_location"] == 1 and has_swedish_loc(row["locations"]):
        return "blue"
    elif row["mentions_location"] == 0 and has_swedish_loc(row["locations"]):
        return "red"


df_exploded["color"] = df_exploded.apply(get_color, axis=1)


geometry = [Point(x, y) for x, y in zip(df_exploded["lon"], df_exploded["lat"])]

# %%
df_exploded["count"] = 1
df_exploded_agg = df_exploded.groupby(["lon", "lat"], as_index=False).agg(
    {"count": "sum", "color": "first", "id": "first", "loc_name": "first"}
)

# %%
df_exploded[df_exploded["loc_name"] == "län"]

# %%
fig = px.scatter_mapbox(
    df_exploded_agg,
    lat="lat",
    lon="lon",
    size="count",
    hover_name=df_exploded_agg["loc_name"],
    hover_data=["id"],
    color_discrete_map={"blue": "blue", "red": "red"},
    color="color",
    mapbox_style="carto-positron",
    height=600,
    zoom=3,
    center={"lat": 63.333112, "lon": 16.007205},
)

fig.show()
