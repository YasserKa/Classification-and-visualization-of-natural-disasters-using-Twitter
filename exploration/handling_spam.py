import ast

import pandas as pd
import visidata
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util

from flood_detection.data.preprocess import Language, Preprocess

# %%
with initialize_config_module(version_base=None, config_module="conf"):
    cfg: DictConfig = compose(config_name="config")

supervisor_proc_geo_path: str = abspath("./" + cfg.supervisor.processed_geo)
df_sup = pd.read_csv(supervisor_proc_geo_path, converters={"user": ast.literal_eval})

api_tweets_path: str = abspath(
    cfg.twitter_api.processed_geo + "_2021-08-17_to_2021-08-23__2023-02-02_22:34:09.csv"
)
df_api = pd.read_csv(api_tweets_path, converters={"user": ast.literal_eval})

df = df_api

# %%

df["created_at"] = pd.to_datetime(df["created_at"])
df_ = df[df["predicted_label"] == 1]

# %%

rows_needed = ["id", "username", "name"]
rows_needed_names = ["user_" + key for key in rows_needed]


def get_values(row):
    return [row[key] for key in rows_needed]


df[rows_needed_names] = df["user"].apply(get_values).tolist()

# %%

# Aggregation over users
df["count"] = 1
df_group = df.groupby(["user_id"], as_index=True)
import json

df_agg = df_group.agg(
    {
        "user_id": "first",
        "user_name": "first",
        "count": "sum",
        "created_at": lambda x: list(x),
        "locations": lambda x: list(
            map(lambda y: list(json.loads(y.replace("'", '"')).keys()), x)
        ),
        # "locations": lambda x: list(map(lambda y: y, x)),
        "id": lambda x: list(x),
    }
)
visidata.vd.view_pandas(df_agg)
# %%

# Use first tweet for each user per week only
df_group = df.groupby(
    [
        "user_id",
        pd.Grouper(
            key="created_at",
            freq="W",
        ),
    ],
    group_keys=True,
)
print(df_group.agg("first"))

# %%

# Using cosine similarity
model = SentenceTransformer("all-MiniLM-L6-v2")


preprocess = Preprocess(language=Language.ENGLISH)
df = df_[:30]
df["text_processed"] = preprocess.clean_text(df["text_translated"].tolist())
df["embeddings"] = df["text_processed"].apply(
    lambda x: model.encode(x, convert_to_tensor=True)
)

x = 0
df["similar"] = 0
for index, row in df.iterrows():
    for index1, row1 in df.iloc[index + 1 :].iterrows():
        cosine_scores = util.cos_sim(row["embeddings"], row1["embeddings"])
        if cosine_scores > 0.8:
            df["similar"][index1] = 1
            print(index)
            print(index1)
            x += 1
            print(cosine_scores)
            print("-----------------------")
            print(row["text_translated"])
            print(row1["text_translated"])
            print("----")
            print(row["text_processed"])
            print(row1["text_processed"])
            print("----")
            print(f"https://twitter.com/anyuser/status/{row['id']}")
            print(f"https://twitter.com/anyuser/status/{row1['id']}")
            print("-----------------------")
print(len(df[df["similar"] != 1]))

# %%
print(df)
