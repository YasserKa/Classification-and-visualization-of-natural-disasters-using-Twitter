#!/usr/bin/env python3

from datetime import datetime

from flood_detection.classes.twitter_facade import TwitterFacade


def main():
    twitter = TwitterFacade()

    query = (
        "(floods OR flood OR översvämning OR översvämningar)" "place_country:SE has:geo"
    )

    tweets = twitter.search_all_tweets(
        query=query,
        start_time=datetime(2021, 8, 18),
        end_time=datetime(2021, 8, 19),
        max_results=10,
    )

    print(tweets)


def extract_from_id(id):

    twitter = TwitterFacade()
    tweets = twitter.get_tweets_from_ids(id)
    print(tweets)


if __name__ == "__main__":
    main()
