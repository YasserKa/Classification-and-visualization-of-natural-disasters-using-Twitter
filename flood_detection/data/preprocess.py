#!/usr/bin/env python3
import ast
import json
import os
import re
import string
import sys
from typing import Union

import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

""" Transform datasets to be ready for training

The data strcture will be:
id, text, text_raw, text_translated, relevant(0/1), mentions_impact(0/1), mentions_location(0/1), other columns provided

Not all datasets include mentions_impact and mentions_location

args:
    1- str: datasetpath
"""


class Preprocess(object):

    # Possible languages en, sv
    def __init__(self, language="en") -> None:
        if language == "sv":
            self.nlp = spacy.load("sv_core_news_sm")
        elif language == "en":
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise Exception(f"{language} is not supported")

    def remove_not_needed_elements_from_string(
        self, text: str, remove_numbers=True
    ) -> str:
        """
        Remove URLs/Mentions/Hashtags/new lines/numbers

        :param text str: to be processed text
        :rtype str: processed text
        """
        regex = (
            "((www.[^s]+)"
            "|(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\."
            "[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))"
            "|(@[a-zA-Z0-9_]*)"
            "|\n"
        )
        if remove_numbers:
            regex += "|([0-9]+)"
        regex += ")"

        processed_text = re.sub(regex, " ", text)

        return processed_text

    def clean_text(self, text_list: Union[list, str]) -> list[str]:
        """
        Clean text of tweet by removing URLs,mentions,hashtag signs,new lines, and numbers

        :param text_list Union[list, str]: A string of text or a list of them
        :rtype list[str]: processed list of strings
        """

        stopwords = self.nlp.Defaults.stop_words

        if isinstance(text_list, str):
            text_list = [text_list]

        for id, text in enumerate(text_list):
            # Lower case
            text = text.lower()

            # Remove URLs/Mentions/Hashtags/new lines/numbers
            text = self.remove_not_needed_elements_from_string(text)
            # Remove punctuation
            text = "".join([char for char in text if char not in string.punctuation])
            # Remove stopwords and very long words
            text = " ".join(
                [
                    word
                    for word in str(text).split()
                    if word not in stopwords and len(word) <= 28
                ]
            )
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
            text_list[id] = text

        return text_list

    def translate_text_list(self, text_list: list[str]):
        return GoogleTranslator(source="auto", target="en").translate_batch(text_list)

    def clean_dataframe(self, df) -> pd.DataFrame:
        """
        Remove retweets, duplicate tweets and clean text

        :param df pd.DataFrame
        """
        print("Cleaning dataframe")
        # Drop tweets that has same text
        df = df[~df["text"].duplicated()]
        # Remove retweets
        df = df.drop(df[df["text"].str.startswith("RT")].index)

        df["text_raw"] = df["text"]
        print("Translating text")
        df["text_translated"] = self.translate_text_list(df["text"].tolist())

        print("Cleaning text")
        # Clean text
        df["text"] = self.clean_text(df["text_translated"].tolist())
        df = df[(df["text"] != "") & df["text"].notnull()]

        return df

    def get_one_tweet_for_each_user_per_week(
        self, df, per_location=False
    ) -> pd.DataFrame:
        """
        Keep only one tweet for each user in a certain week (optionally for each location)
        Used to reduce spammers
                 df Dataframe
                 per_location boolean

             Returns:
                 Dataframe
        """
        df["user"] = df["user"].apply(lambda x: ast.literal_eval(x))
        df["user_id"] = df["user"].apply(lambda x: x["id"])

        df["created_at"] = pd.to_datetime(df["created_at"])
        group_by_keys = [
            "user_id",
            pd.Grouper(key="created_at", freq="1W"),
        ]
        if per_location:
            group_by_keys += ["loc_name"]

        # Use first tweet for each user per week only
        df_group = df.groupby(
            group_by_keys,
            group_keys=False,
        )
        df_user_week_uniq = df_group.apply(lambda x: x)
        df_user_week_uniq.reset_index(inplace=True)
        return df_user_week_uniq


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
    df = df.rename(
        columns={
            "tweet_text": "text",
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
        columns={"tweet_id": "id", "tweet": "text", "label": "relevant"},
    )
    df["relevant"] = df["relevant"].apply(lambda x: 1 if x == "on-topic" else 0)
    df["id"] = df["id"].apply(lambda x: x[1:-1])
    df = df.astype({"id": "int", "relevant": "int"})
    return df


def preprocess_tweets(path) -> pd.DataFrame:
    with open(path, "r") as file:
        tweets_json = json.load(file)

    df = pd.json_normalize(list(tweets_json.values()), max_level=0)

    df = df.astype({"id": "int"})
    return df


def preprocess_supervisor_dataset(path) -> pd.DataFrame:
    with open(path, "r") as file:
        tweets_json = json.load(file)

    df = pd.json_normalize(list(tweets_json.values()), max_level=0)

    df = df.rename(
        columns={
            "On Topic": "relevant",
            "Explicit location in Sweden": "mentions_location",
            "Contains specific information about IMPACTS": "mentions_impact",
        }
    )
    df = df[
        (df["relevant"] != "")
        & (df["mentions_impact"] != "")
        & (df["mentions_location"] != "")
    ]
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

    path: str = abspath(sys.argv[1])

    if path == abspath(cfg.supervisor.tweets):
        output: str = abspath("./" + cfg.supervisor.processed)
        df = preprocess_supervisor_dataset(path)
    elif path.startswith(abspath(cfg.twitter_api.tweets)):
        # Use csv extension instead of json
        output_file_name = os.path.basename(path).split(".")[0] + ".csv"
        output: str = abspath(
            "./" + os.path.dirname(cfg.twitter_api.processed) + "/" + output_file_name
        )
        df = preprocess_tweets(path)
    elif path == abspath(cfg.alberta.raw):
        output: str = abspath("./" + cfg.alberta.processed)
        df = preprocess_crisislex_dataset(path)
    elif path == abspath(cfg.queensland.raw):
        output: str = abspath("./" + cfg.queensland.processed)
        df = preprocess_crisislex_dataset(path)
    elif path == abspath(cfg.crisis.raw):
        output: str = abspath("./" + cfg.crisis.processed)
        df = preprocess_crisis_dataset(path)
    else:
        raise Exception(f"{path} file not found")

    preprocess = Preprocess()
    df = preprocess.clean_dataframe(df)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
