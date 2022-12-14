#!/usr/bin/env python3
import csv
import json

from deep_translator import GoogleTranslator
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

from flood_detection.classes.twitter_facade import TwitterFacade


# REFACTOR: if this going to be used often, refactor it, and include it in the pipeline
def extract_tweets_for_supervisor_data() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")
        supervisor_raw_path: str = abspath(cfg.supervisor.raw)
        supervisor_tweets_path: str = abspath(cfg.supervisor.tweets)
    twitter: TwitterFacade = TwitterFacade()

    tweets = {}

    with open(supervisor_raw_path, newline="") as csvfile:
        rows = list(csv.DictReader(csvfile, delimiter=","))
        tweets_dict = dict((int(row["id"]), row) for row in rows)

        tweets = twitter.get_tweets_from_ids(list(tweets_dict.keys()))

        # Add the labels to the tweets
        for id in tweets.keys():
            tweets[id].update(tweets_dict[id])

    # Write pretty print JSON data to file
    with open(supervisor_tweets_path, "w") as outfile:
        json.dump(tweets, outfile, indent=4, sort_keys=True, default=str)


if __name__ == "__main__":
    extract_tweets_for_supervisor_data()
