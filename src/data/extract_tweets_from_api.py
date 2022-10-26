#!/usr/bin/env python3

import json
from datetime import datetime

import click
from deep_translator import GoogleTranslator
from hydra import compose, initialize
from omegaconf import DictConfig

from src.classes.twitter_facade import TwitterFacade


@click.command()
@click.option(
    "--from",
    "from_date",
    required=True,
    help="Extract tweets from this date YYYY-MM-DD format",
)
@click.option(
    "--to",
    "to_date",
    required=True,
    help="Extract tweets till this date YYYY-MM-DD format",
)
def main(to_date, from_date) -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")

    output_path: str = cfg.twitter_api.tweets
    twitter: TwitterFacade = TwitterFacade()
    to_date = list(map(lambda x: int(x), to_date.split("-")))
    from_date = list(map(lambda x: int(x), from_date.split("-")))

    # query: str = (
    #     "place_country:SE"
    # )
    # query: str = (
    #     "place_country:SE lang:sv"
    # )

    query: str = (
        '"atmosfärisk flod" OR "hög vatten" OR åskskur'
        ' OR regnskur OR dagvattensystem OR dränering OR "höga vågor"'
        ' OR "höga flöden" OR dämmor'
        " OR snösmältning OR blött OR oväder OR stormflod OR vattenstånd"
        " OR vattennivå OR åskväder OR regnstorm"
        ' OR "mycket regn" OR "kraftig regn" OR översvämningsskador OR översvämningar OR översvämning'
    )

    tweets = twitter.search_all_tweets(
        query=query,
        start_time=datetime(*from_date),
        end_time=datetime(*to_date),
        tweet_fields=twitter.TWEET_FIELDS_EXTRACTED,
        place_fields=twitter.PLACE_FIELDS,
        expansions="geo.place_id",
        max_results=10,
    )

    # Add the labels to the tweets
    for id in tweets.keys():
        tweets[id].update(tweets[id])
        tweets[id]["text_en"] = GoogleTranslator(source="auto", target="en").translate(
            tweets[id]["text"]
        )

    with open(output_path, "w") as outfile:
        json.dump(tweets, outfile, indent=2)


if __name__ == "__main__":
    main()
