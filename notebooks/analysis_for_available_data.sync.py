# %%
import json

import matplotlib.pyplot as plt
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
df = pd.json_normalize(list(tweets_json.values()))

# %%
df.head()


# %%
def value_counts_lambda(col):
    return print(df[col].value_counts())


value_counts_lambda("On Topic")
value_counts_lambda("Informative/relevant/non sarcastic")
value_counts_lambda("Contains specific information about IMPACTS")

# %%
df = df[
    (df["On Topic"] != "")
    & (df["Informative/relevant/non sarcastic"] != "")
    & (df["Contains specific information about IMPACTS"] != "")
    & (df["Explicit location in Sweden"] != "")
]
df["created_at"] = pd.to_datetime(df["created_at"])

df["On Topic"] = df["On Topic"].astype(int)
df["Explicit location in Sweden"] = df["Explicit location in Sweden"].astype(int)

df["Informative/relevant/non sarcastic"] = df[
    "Informative/relevant/non sarcastic"
].astype(int)
df["Contains specific information about IMPACTS"] = df[
    "Contains specific information about IMPACTS"
].astype(int)

df_gb = df.groupby([df["created_at"].dt.date])

# %%

df_gb = df.groupby([pd.Grouper(key="created_at", freq="W")])

# %%
fig, ax1 = plt.subplots()
dates = df_gb.groups.keys()

plt.plot(dates, df_gb["On Topic"].sum().values)
plt.plot(dates, df_gb["Informative/relevant/non sarcastic"].sum().values)
plt.plot(dates, df_gb["Contains specific information about IMPACTS"].sum().values)

plt.title("Time series for tweets")
plt.xlabel("Date")
plt.ylabel("count")

plt.show()

# %%

df_gb["On Topic"].sum().sort_values()


# %%

print(df_gb.get_group("2015-07-12 00:00:00+00:00").sort_values(by="created_at"))

# %%

df_ = df[
    (df["created_at"] >= "2021-08-18")
    & (df["created_at"] <= "2021-08-23")
    & (df["Explicit location in Sweden"] == 1)
    & (df["On Topic"] == 1)
]
# df_
visidata.vd.view_pandas(df_)
