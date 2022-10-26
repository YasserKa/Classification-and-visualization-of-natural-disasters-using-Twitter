#!/usr/bin/env python3

from enum import Enum

import click
import pandas as pd
from geopy.geocoders import GeoNames, Nominatim
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import pipeline

from src.data.preprocess import remove_not_needed_elements_from_string

"""
Extract locations from swedish text
"""


# Check the link below for compelte list of geocoders
# https://geopy.readthedocs.io/en/stable/#geonames
class GeoCodersEnum(Enum):
    NOMINATIM = "nominatim"
    GEONAMES = "geonames"


class Transform:
    def __init__(self, name):
        if name not in [
            "KBLab/bert-base-swedish-cased-ner",
            "KB/bert-base-swedish-cased-ner",
        ]:
            raise Exception(f"{name} is not supported")
        self.__initialize_pipeline(name)

    def __initialize_pipeline(self, name):
        self.__nlp = pipeline(
            "ner",
            model=name,
            tokenizer=name,
            # Default ignore list is ["O"] which are tokens needed to extract locations
            ignore_labels=[],
        )

    def get_tokens(self, text: str):
        text = remove_not_needed_elements_from_string(text, remove_numbers=False)
        tokens = self.__nlp(text)
        merged_tokens = []
        for token in tokens:
            if token["word"].startswith("##"):
                merged_tokens[-1]["word"] += token["word"][2:]
            else:
                merged_tokens += [token]
        # Remove not needed tokens
        return list(filter(lambda x: x["entity"] != "O", merged_tokens))

    def get_location_tokens(self, tokens: list[dict]) -> dict[str, dict[str, float]]:
        loc_tokens: dict[str, dict[str, float]] = {}
        for token in tokens:
            if token["entity"] == "LOC":
                loc_tokens[token["word"]] = {"ner_score": token["score"]}
        return loc_tokens


class GeoCoder:
    """Geocoder to extract information about locations from text"""

    def __init__(self, geocoder_name):
        self.__geocoder_name = GeoCodersEnum(geocoder_name)
        match self.__geocoder_name:
            case GeoCodersEnum.NOMINATIM:
                self.__geocoder_obj: Nominatim = Nominatim(user_agent="flood_detection")
            case GeoCodersEnum.GEONAMES:
                self.__geocoder_obj: GeoNames = GeoNames("yasser_kaddoura")
            case _:
                raise Exception(f"{self.__geocoder_name} geocoder isn't available")

    def __get_location(self, text):
        match self.__geocoder_name:
            case GeoCodersEnum.NOMINATIM:
                loc = self.__geocoder_obj.geocode(
                    text, country_codes="se", language="en", extratags=True
                )
            case GeoCodersEnum.GEONAMES:
                loc = self.__geocoder_obj.geocode(text, country="SE", country_bias="SE")
                pass
            case _:
                raise Exception(f"{self.__geocoder_name} geocoder isn't available")
        return loc

    def get_swedish_location(self, locations: dict[str, dict]) -> dict[str, dict]:
        swedish_locations = locations.copy()
        for name in locations:
            swedish_location = self.__get_location(name)

            if swedish_location is not None:
                swedish_locations[name]["swedish_loc_info"] = swedish_location.raw
            else:
                swedish_locations[name]["swedish_loc_info"] = {}

        return swedish_locations


@click.command()
@click.option(
    "--model_name",
    default="KBLab/bert-base-swedish-cased-ner",
    help="Model used to train",
)
@click.option(
    "--geocoder_name",
    default=GeoCodersEnum.NOMINATIM.value,
    help="Gazetteer used to extract info about locations",
)
@click.argument("path_to_data", nargs=-1)
def main(model_name, geocoder_name, path_to_data):
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")

    input_path: str = path_to_data[0]
    df = pd.read_csv(input_path)

    if input_path == cfg.supervisor.processed_flood:
        output_path: str = cfg.supervisor.processed_geo
        df = df[df["predicted_label"] == 1]
    elif input_path == cfg.twitter_api.processed_flood:
        output_path = cfg.twitter_api.processed_geo
        df = df[df["predicted_label"] == 1]

    model = Transform(model_name)
    geocoder = GeoCoder(geocoder_name)

    tqdm.pandas(desc="NER NLP")
    df["tokens"] = df["raw_text"].progress_apply(model.get_tokens)
    df["locations"] = df["tokens"].apply(model.get_location_tokens)
    tqdm.pandas(desc="Swedish locations")
    df["locations"] = df["locations"].progress_apply(geocoder.get_swedish_location)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
