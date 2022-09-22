#!/usr/bin/env python3
import math
import os

import numpy as np
import tweepy
from dotenv import load_dotenv


class TwitterFacade:
    """
    Class that interacts with twitter API
    """

    # Tweets fields
    # https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
    TWEET_FIELDS_AVAILABLE = [
        "attachments",
        "author_id",
        "context_annotations",
        "conversation_id",
        "created_at",
        "entities",
        "geo",
        "id",
        "in_reply_to_user_id",
        "lang",
        "non_public_metrics",
        "organic_metrics",
        "possibly_sensitive",
        "promoted_metrics",
        "public_metrics",
        "referenced_tweets",
        "reply_settings",
        "source",
        "text",
        "withheld",
    ]

    TWEET_FIELDS_EXTRACTED = [
        "id",
        "text",
        "public_metrics",
        "referenced_tweets",
        "author_id",
        "attachments",
        "geo",
        "source",
        "lang",
        "created_at",
    ]

    PLACE_FIELDS = [
        "contained_within",
        "country",
        "country_code",
        "full_name",
        "geo",
        "id",
        "name",
        "place_type",
    ]

    def __init__(self) -> None:
        load_dotenv()
        self.api = tweepy.Client(os.environ.get("BEARER_TOKEN"))

    def parse_twitter_response(self, twitter_respose):
        tweets_dict = {}
        places = {}

        if "places" in twitter_respose.includes:
            for place in twitter_respose.includes["places"]:
                places[place.data["id"]] = place.data

        for tweet in twitter_respose.data:
            id = tweet["id"]
            tweets_dict[id] = tweet.data
            if "geo" in tweet:
                tweets_dict[id]["place"] = places[tweet["geo"]["place_id"]]
                del tweets_dict[id]["geo"]

        return tweets_dict

    def get_tweets_from_id(self, ids: list[int]):
        tweets = dict()

        #  Twitter API accepts 100 tweets per request, so they are seperated
        #  in a list of lists that have 100 tweet ids at most
        list_of_list_of_ids = np.array_split(ids, math.ceil(len(ids) / 100))

        for list_of_ids in list_of_list_of_ids:
            twitter_respose = self.api.get_tweets(
                ids=",".join(map(str, list_of_ids)),
                expansions="geo.place_id",
                place_fields=self.PLACE_FIELDS,
                tweet_fields=self.TWEET_FIELDS_EXTRACTED,
            )

            tweets.update(self.parse_twitter_response(twitter_respose))

        return tweets
