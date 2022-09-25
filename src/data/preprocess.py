#!/usr/bin/env python3
import json
import re
import string
import sys
from typing import Union

import pandas as pd
import spacy
from hydra import compose, initialize
from omegaconf import DictConfig

from src.data.text_processing import TextProcessing

""" Transform datasets to be ready for training

The data strcture will be:

tweet_id, text, label

label: (0/1) on topic or not
"""


def clean_text(text_list: Union[list, str]) -> list[str]:
    """
    Clean text of tweet by removing URLs,mentions,hashtag signs,new lines, and numbers

    :param text_list Union[list, str]: A string of text or a list of them
    :rtype list[str]: processed list of strings
    """

    sp = spacy.load("en_core_web_sm")
    stopwords = sp.Defaults.stop_words

    if isinstance(text_list, str):
        text_list = [text_list]

    for id, text in enumerate(text_list):

        # Remove URLs/Mentions/Hashtags/new lines/numbers
        text = re.sub(
            "((www.[^s]+)"
            "|(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\."
            "[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))"
            "|(@[a-zA-Z0-9]*)"
            "|([0-9]+)"
            "|\n)",
            " ",
            text,
        )
        # Remove stopwords
        text = " ".join([word for word in str(text).split() if word not in stopwords])
        # Remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])
        # Remove Emojis
        emoji_pattern: re.Pattern[str] = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            # flags (iOS)
            "\U0001F1E0-\U0001F1FF" "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)
        # Lower case
        text = text.lower()
        text_list[id] = text

    return text_list


def clean_dataframe(df) -> pd.DataFrame:
    """
    Remove retweets, duplicate tweets and clean text

    :param df [TODO:type]: [TODO:description]
    """
    # Remove retweets
    df = df[df["text"].str[:2] != "RT"]
    # Remove duplicates
    df = df.drop_duplicates()

    text_preprocessing = TextProcessing()
    # Clean text
    df["text"] = text_preprocessing.clean_text(df["text"].to_numpy())
    return df


def transform_crisislex_dataset(path) -> pd.DataFrame:
    """
    [TODO:description]

    :param path [TODO:type]: [TODO:description]
    :rtype pd.DataFrame: [TODO:description]
    """
    df: pd.DataFrame = pd.read_csv(path)
    df = df.rename(columns={"tweet_id": "id", "tweet": "text"})
    df["label"] = df["label"].apply(lambda x: 1 if x == "on-topic" else 0)
    df["id"] = df["id"].apply(lambda x: x[1:-1])
    df = df.astype({"id": "int"})
    df = clean_dataframe(df)
    return df


def transform_supervisor_dataset(path) -> pd.DataFrame:
    with open(path, "r") as file:
        tweets_json = json.load(file)

    df = pd.json_normalize(list(tweets_json.values()))
    df = df[["id", "text_en", "On Topic"]]
    df = df.rename(columns={"text_en": "text", "On Topic": "label"})
    df = df[df["label"] != ""]
    df = df.astype({"label": "int", "id": "int"})
    df = clean_dataframe(df)
    return df


def main() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")

    path: str = sys.argv[1]
    match path:
        case cfg.supervisor.tweets:
            output: str = cfg.supervisor.processed
            df = transform_supervisor_dataset(path)
        case cfg.alberta.raw:
            output: str = cfg.alberta.processed
            df = transform_crisislex_dataset(path)
        case cfg.queensland.raw:
            output: str = cfg.queensland.processed
            df = transform_crisislex_dataset(path)
        case _:
            raise Exception(f"{path} file not found")

    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
