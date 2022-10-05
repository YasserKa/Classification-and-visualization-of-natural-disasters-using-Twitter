# %%
from transformers import AutoModel, AutoTokenizer
from hydra import compose, initialize
from omegaconf import DictConfig
from src.data.preprocess import remove_not_needed_elements_from_string
from transformers import pipeline
from tqdm.notebook import tqdm
import pandas as pd
from geopy.geocoders import Nominatim

# %%
with initialize(version_base=None, config_path="../conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed

df = pd.read_csv("../" + path_to_data)

print(
    f"Number of tweets that explicity mentions locations\n{len(df[df['mentions_location'] == 1])}/{len(df)}"
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
geolocator = Nominatim(user_agent="flood_detection")


def is_swedish_geo(list_entities):
    for geo in list_entities:
        entity_name = list(geo.keys())[0]
        swedish_location = geolocator.geocode(
            entity_name, country_codes="se", language="en"
        )
        if swedish_location is not None:
            return True
    return False


df["has_loc_entities"] = df["tokens"].apply(lambda x: len(x) > 0)
tqdm.pandas(desc="Is Swidsh")
df["loc_ent_is_swedish"] = df["tokens"].progress_apply(is_swedish_geo)

# %%
confusion_matrix = pd.crosstab(
    df["mentions_location"],
    df["loc_ent_is_swedish"],
    rownames=["Actual"],
    colnames=["Predicted"],
)
confusion_matrix

# %% [markdown]
# The percision seems to be not that good (i.e. The model seems to predict that some locations are swedish, but they are not)

# %%
469 / (469 + 218)

# %%
df[(df["mentions_location"] == 0) & df["loc_ent_is_swedish"]][["id", "tokens"]].head(10)

# %%
swedish_location = geolocator.geocode(
    "Florida", country_codes="se", language="en", extratags=True, addressdetails=True
)
swedish_location.raw

# %%
swedish_location = geolocator.geocode(
    "Stockholm", country_codes="se", language="en", extratags=True, addressdetails=True
)
swedish_location.raw["extratags"]


# %% [markdown]
# It seems that some locations in Sweden use terms like miami and Florida.
# Let's update the filter so that we limit them further. Importance and population seem to be logical picks.
# Let's filter out places that don't have population tag or have less than 1000

# %%
def is_swedish_geo(list_entities):
    for geo in list_entities:
        entity_name = list(geo.keys())[0]
        swedish_location = geolocator.geocode(
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
df[df["loc_ent_is_swedish"] != "False"]["loc_ent_is_swedish"]

# %%
swedish_location = geolocator.geocode(
    "Norrsundet", country_codes="se", language="en", extratags=True, addressdetails=True
)
swedish_location.raw
