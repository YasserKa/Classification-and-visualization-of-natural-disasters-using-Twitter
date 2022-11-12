import ast
import json
import sys
from pprint import pprint

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig

# %%

with initialize(version_base=None, config_path="conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed_geo

df = pd.read_csv(path_to_data, converters={"locations": ast.literal_eval})

# %%
data_needed = [
    "class",
    "importance",
    "type",
    "display_name",
    "lat",
    "lon",
    "boundingbox",
]


def get_from_raw_loc(row):
    locations = {}
    for name, value in row.items():
        if len(value["swedish_loc_info"]) > 0:
            location_info = {
                data: value["swedish_loc_info"][data] for data in data_needed
            }
            locations[name] = location_info
    return locations


df["locations_info"] = df["locations"].apply(get_from_raw_loc)
# df["locations_info"] = df["locations_info"].apply(lambda x: list(x.items()))


# %%
def get_location_with_lowest_param(row):
    lowest_param = 999
    curr_location = {}
    for location in row.values():
        try:
            bounding_box = location["boundingbox"]
            param = abs(float(bounding_box[0]) - float(bounding_box[1])) + abs(
                float(bounding_box[2]) - float(bounding_box[3])
            )
            if param < lowest_param:
                lowest_param = param
                curr_location = location
        except Exception:
            print(row)

    return curr_location


df["locations_info"].apply(get_location_with_lowest_param)
