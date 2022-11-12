#!/usr/bin/env python3

import json
from datetime import datetime

import click
from hydra import compose, initialize
from omegaconf import DictConfig

from flood_detection.classes.twitter_facade import TwitterFacade


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

    twitter: TwitterFacade = TwitterFacade()

    output_path: str = f"{cfg.paths.tweets}/twitter_api_{from_date}_to_{to_date}.json"
    to_date = list(map(lambda x: int(x), to_date.split("-")))
    from_date = list(map(lambda x: int(x), from_date.split("-")))

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
    )

    with open(output_path, "w") as outfile:
        json.dump(tweets, outfile, indent=4, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
