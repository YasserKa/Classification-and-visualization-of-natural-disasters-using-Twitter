#!/usr/bin/env python3

import json
from datetime import datetime

from src.classes.twitter_facade import TwitterFacade


def main():
    twitter = TwitterFacade()

    query = (
        "(floods OR flood OR översvämning OR översvämningar)" "place_country:SE has:geo"
    )

    tweets = twitter.search_all_tweets(
        query=query,
        start_time=datetime(2021, 8, 17),
        end_time=datetime(2021, 8, 19),
        tweet_fields=twitter.TWEET_FIELDS_EXTRACTED,
        place_fields=twitter.PLACE_FIELDS,
        expansions="geo.place_id",
        max_results=10,
    )

    with open("data.json", "w") as outfile:
        json.dump(tweets, outfile, indent=2)

    return tweets


if __name__ == "__main__":
    main()
