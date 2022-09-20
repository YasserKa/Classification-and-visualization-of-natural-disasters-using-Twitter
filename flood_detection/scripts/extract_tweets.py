#!/usr/bin/env python3
from flood_detection.twitter.TwitterAdapter import TwitterAdapter
from datetime import datetime
import json
import sys


def main():
    twitter = TwitterAdapter()

    query = ("(floods OR flood OR översvämning OR översvämningar)"
             "place_country:SE has:geo")

    twitter_respose = twitter.api.search_all_tweets(
        query=query,
        start_time=datetime(2021, 8, 17),
        end_time=datetime(2021, 8, 19),
        tweet_fields=twitter.TWEET_FIELDS_EXTRACTED,
        place_fields=twitter.PLACE_FIELDS,
        expansions="geo.place_id",
        max_results=10)

    tweets = twitter.parse_twitter_response(twitter_respose)

    # Write pretty print JSON data to file
    with open(f"{twitter.OUTPUT_PATH}/tweets.json", "w") as write_file:
        json.dump(tweets, write_file, indent=4)


if __name__ == '__main__':
    sys.exit(main())
