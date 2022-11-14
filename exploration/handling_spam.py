import ast

import pandas as pd
import visidata
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

# %%

with initialize_config_module(version_base=None, config_module="conf"):
    cfg: DictConfig = compose(config_name="config")

supervisor_proc_geo_path: str = abspath("./" + cfg.supervisor.processed_geo)
df_sup = pd.read_csv(supervisor_proc_geo_path, converters={"user": ast.literal_eval})

api_tweets_path: str = abspath(
    cfg.twitter_api.processed_geo + "_2021-08-17_to_2021-08-23.csv"
)
df_api = pd.read_csv(api_tweets_path, converters={"user": ast.literal_eval})

df_sup = df_api


# %%

df_sup["created_at"] = pd.to_datetime(df_sup["created_at"])
df_sup = df_sup[df_sup["predicted_label"] == 1]

# %%

rows_needed = ["id", "username", "name"]
rows_needed_names = ["user_" + key for key in rows_needed]


def get_values(row):
    return [row[key] for key in rows_needed]


df_sup[rows_needed_names] = df_sup["user"].apply(get_values).tolist()
# visidata.vd.view_pandas(x)
# {', 'end': 122, 'expanded_url': 'http://dn.se/dn-i-sociala-m', 'start': 99, 'url': 'https://t.co/oR28flGF0h'}]}, 'url': {'urls': [{'display_url': 'dn.se', 'end': 23, 'expanded_url': 'http://www.dn.se', 'start': 0, 'url': 'https://t.co/HnfyqYX0Ko'}]}}, 'id': '592490041', 'name': 'Dagens Nyheter', 'profile_image_url': 'https://pbs.twimg.com/profile_images/1162358512535707650/AW8XtZqx_normal.jpg', 'protected': False, 'url': 'https://t.co/HnfyqYX0Ko', 'username': 'dagensnyheter', 'verified': True}

# %%

df_sup["count"] = 1
df_group = df_sup.groupby(["user_id"], as_index=True)
df_agg = df_group.agg(
    {
        "user_id": "first",
        "user_name": "first",
        "count": "sum",
        "created_at": lambda x: list(x),
        "locations": lambda x: list(x),
        "id": lambda x: list(x),
    }
)
visidata.vd.view_pandas(df_agg)
# %%

df_users = df_sup[df_sup["user_id"].isin(["912854161", "898442239"])]
# Use first tweet for each user per week only
df_group = df_sup.groupby(
    [
        "user_id",
        pd.Grouper(
            key="created_at",
            freq="W",
        ),
    ],
    group_keys=True,
)
print(df_group.agg({"id": "first"}))
print(len(df_sup))

# %%
# Username: sverigevarning -> Sweden Warning
# https://twitter.com/sverigevarning
# Not sure if this bot is affiliated with SMHI
