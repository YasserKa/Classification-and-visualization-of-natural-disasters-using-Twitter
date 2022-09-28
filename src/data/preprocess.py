#!/usr/bin/env python3
import json
import re
import string
import sys
from typing import Union

import pandas as pd
import numpy as np
import spacy
from hydra import compose, initialize
from omegaconf import DictConfig

""" Transform datasets to be ready for training

The data strcture will be:
id, text, relevant(0/1), mentions_impact(0/1)

Not all datasets include "mentions_impact"

args:
    1- str: datasetpath
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

    :param df pd.DataFrame
    """
    # Remove retweets
    df = df[df["text"].str[:2] != "RT"]
    # Remove duplicates
    df = df.drop_duplicates()

    # Clean text
    df["text"] = clean_text(df["text"].to_numpy())
    df = df[(df["text"] != "") & df["text"].notnull()]

    return df


def preprocess_crisislex_dataset(path) -> pd.DataFrame:
    """
    Preprocess a crisislex dataset
    Its structure is the following:
    tweet_id, tweet, label

    :param path str
    :rtype pd.DataFrame
    """
    df: pd.DataFrame = pd.read_csv(path)
    df = df.rename(
        columns={"tweet_id": "id", "tweet": "text", "label": "relevant"},
    )
    df["relevant"] = df["relevant"].apply(lambda x: 1 if x == "on-topic" else 0)
    df["id"] = df["id"].apply(lambda x: x[1:-1])
    df = df.astype({"id": "int"})
    df = clean_dataframe(df)
    return df


def preprocess_supervisor_dataset(path) -> pd.DataFrame:
    with open(path, "r") as file:
        tweets_json = json.load(file)

    df = pd.json_normalize(list(tweets_json.values()))
    df = df[
        ["id", "text_en", "On Topic", "Contains specific information about IMPACTS"]
    ]
    df = df.rename(
        columns={
            "text_en": "text",
            "On Topic": "relevant",
            "Contains specific information about IMPACTS": "mentions_impact",
        }
    )
    df = df[(df["relevant"] != "") & (df["mentions_impact"] != "")]
    df = df.astype({"relevant": "int", "mentions_impact": "int", "id": "int"})
    df = clean_dataframe(df)
    return df


def main() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")

    path: str = sys.argv[1]
    match path:
        case cfg.supervisor.tweets:
            output: str = cfg.supervisor.processed
            df = preprocess_supervisor_dataset(path)
        case cfg.alberta.raw:
            output: str = cfg.alberta.processed
            df = preprocess_crisislex_dataset(path)
        case cfg.queensland.raw:
            output: str = cfg.queensland.processed
            df = preprocess_crisislex_dataset(path)
        case _:
            raise Exception(f"{path} file not found")

    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
