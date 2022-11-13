# %%
import json

import pandas as pd
import visidata
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

# %%

with initialize(version_base=None, config_path="conf"):
    cfg: DictConfig = compose(config_name="config")


supervisor_tweets_path: str = abspath(
    "./" + cfg.supervisor.tweets,
)

with open(supervisor_tweets_path, "r") as file:
    tweets_json = json.load(file)
    df_supervisor = pd.json_normalize(list(tweets_json.values()))


api_tweets_path: str = abspath(
    "./" + cfg.twitter_api.processed_geo + "_2021-08-17_to_2021-08-23.csv"
)

df_api = pd.read_csv(api_tweets_path)


# %%
df_supervisor["created_at"] = pd.to_datetime(df_supervisor["created_at"])
df_supervisor = df_supervisor[
    (df_supervisor["On Topic"] != "")
    & (df_supervisor["Informative/relevant/non sarcastic"] != "")
    & (df_supervisor["Contains specific information about IMPACTS"] != "")
    & (df_supervisor["Explicit location in Sweden"] != "")
]

df_supervisor = df_supervisor.astype(
    {"On Topic": "int", "Explicit location in Sweden": "int"}
)

# %%

# NOTES: There are tweets only for the 19th in that week in that time interval
# 8 rows
df_18_to_23_loc = df_supervisor[
    (df_supervisor["created_at"] >= "2021-08-18")
    & (df_supervisor["created_at"] <= "2021-08-23")
    & (df_supervisor["Explicit location in Sweden"] == 1)
    & (df_supervisor["On Topic"] == 1)
]

# 48 rows
df_18_to_23 = df_supervisor[
    (df_supervisor["created_at"] >= "2021-08-18")
    & (df_supervisor["created_at"] <= "2021-08-23")
    & (df_supervisor["On Topic"] == 1)
]

# %%

visidata.vd.view_pandas(df_18_to_23)
