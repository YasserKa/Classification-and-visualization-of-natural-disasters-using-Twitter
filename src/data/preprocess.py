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

""" Transform datasets to be ready for training

The data strcture will be:
id, text, raw_text, relevant(0/1), mentions_impact(0/1), mentions_location(0/1)

Not all datasets include mentions_impact and mentions_location

args:
    1- str: datasetpath
"""


def remove_not_needed_elements_from_string(text: str) -> str:
    """
    Remove URLs/Mentions/Hashtags/new lines/numbers

    :param text str: to be processed text
    :rtype str: processed text
    """

    processed_text = re.sub(
        "((www.[^s]+)"
        "|(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\."
        "[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))"
        "|(@[a-zA-Z0-9_]*)"
        "|([0-9]+)"
        "|\n)",
        " ",
        text,
    )

    return processed_text


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
        text = remove_not_needed_elements_from_string(text)
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
    # HACK: process the english translated tweets instead of the original
    # swedish ones
    # TODO: include the swe->en translation step here by detecting the language
    # of the raw text and adding it as a new column, and translate it
    if "text" in df.columns:
        raw_text_series = df["text"]
    else:
        raw_text_series = df["raw_text"]

    # Remove retweets
    df = df.drop(df[raw_text_series.str.startswith("RT")].index)
    # Remove duplicates
    df = df.drop_duplicates()

    if "text" in df.columns:
        raw_text_series = df["text"]
    else:
        raw_text_series = df["raw_text"]

    # Clean text
    df["text"] = clean_text(raw_text_series.to_numpy())
    df = df[(df["text"] != "") & df["text"].notnull()]

    return df


def preprocess_crisis_dataset(path) -> pd.DataFrame:
    """
     Preprocess a crisis dataset to get impact
     Its structure is the following:
     event_name	tweet_id	image_id	tweet_text	image	label

     Label shows the impact severity:
    * Severe damage
    * Mild damage
    * Little or no damage

    mentions_impact is 1 if the severity is (Mild or Sever), 0 otherwise

     :param path str
     :rtype pd.DataFrame
    """
    df: pd.DataFrame = pd.read_csv(path, sep="\t")
    df = df[["tweet_id", "event_name", "tweet_text", "label"]]
    df = df.rename(
        columns={
            "tweet_text": "raw_text",
            "tweet_id": "id",
            "label": "mentions_impact",
        }
    )
    impactful_labels = ["sever_damage", "mild_damage"]
    df["mentions_impact"] = df["mentions_impact"].apply(
        lambda x: 1 if x in impactful_labels else 0
    )
    df = df.astype({"id": "int", "mentions_impact": "int"})
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
        columns={"tweet_id": "id", "tweet": "raw_text", "label": "relevant"},
    )
    df["relevant"] = df["relevant"].apply(lambda x: 1 if x == "on-topic" else 0)
    df["id"] = df["id"].apply(lambda x: x[1:-1])
    df = df.astype({"id": "int", "relevant": "int"})
    return df


def preprocess_supervisor_dataset(path) -> pd.DataFrame:
    with open(path, "r") as file:
        tweets_json = json.load(file)

    df = pd.json_normalize(list(tweets_json.values()))
    # HACK: check line 86
    df = df[
        [
            "id",
            "text",
            "Explicit location in Sweden",
            "text_en",
            "On Topic",
            "Contains specific information about IMPACTS",
        ]
    ]
    df = df.rename(
        columns={
            "text_en": "text",
            "text": "raw_text",
            "On Topic": "relevant",
            "Explicit location in Sweden": "mentions_location",
            "Contains specific information about IMPACTS": "mentions_impact",
        }
    )
    df = df[(df["relevant"] != "") & (df["mentions_impact"] != "")]
    df = df.astype(
        {
            "relevant": "int",
            "mentions_impact": "int",
            "mentions_location": "int",
            "id": "int",
        }
    )
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
        case cfg.crisis.raw:
            output: str = cfg.crisis.processed
            df = preprocess_crisis_dataset(path)
        case _:
            raise Exception(f"{path} file not found")

    df = clean_dataframe(df)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
