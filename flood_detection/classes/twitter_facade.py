#!/usr/bin/env python3

import math
import os
from time import sleep

import numpy as np
import tweepy
from dotenv import load_dotenv
from tweepy.errors import BadRequest, TooManyRequests


class TwitterFacade:
    """
    Class that interacts with twitter API via tweepy
    """

    # Tweets fields
    # https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
    TWEET_FIELDS = [
        "attachments",
        "author_id",
        "context_annotations",  # entity recognition/extraction, topical analysis
        "edit_history_tweet_ids",
        "edit_controls",  # Number of remaining edits (max is 5)
        "conversation_id",  # The original tweet of the conversaion
        "created_at",
        "entities",  # Entities (urls, hashtagsh, annotations)
        "in_reply_to_user_id",  # Used to indicate if this a reply or not
        "geo",  # Contains place details
        "id",
        "lang",  # Language detect by twitter
        "public_metrics",  # Counts for likes, retweets, replies, quotes
        "referenced_tweets",  # For reteweets Mentions if the parent tweet is replied_to or quoted
        "reply_settings",  # Who can reply to the tweet, everyone, mentioned_users, followers
        "source",  # From where the tweet was posted from
        "text",
        "withheld",  # If the tweet is witheld
    ]
    USER_FIELDS = [
        "created_at",
        "description",  # Profile description
        "entities",  # Entities in description
        "id",
        "location",  # Specified in user's profile if any (might not be valid)
        "name",
        "pinned_tweet_id",  # Can potentially determine user's langauge
        "profile_image_url",  # Can be usd to download the image
        "protected",  # If tweets are private
        "url",  # URL used in profile if any
        "username",  # Alias, Unique
        "verified",  # A verified account lets people know that an account of public interest is authentic.
        "withheld",
    ]

    MEDIA_FIELDS = [
        "alt_text",  # description
        "duration_ms",
        "height",  # height of content in pixels
        "media_key",  # Used to programmatically retrieve media
        "preview_image_url",  # Static placeholder preview
        "public_metrics",  # view count
        "type",  # photo, GIF, or video
        "url",  # Media oject with URL field for photos
        "variants",  # List of display, or playback variants
        "width",  # width of content in pixels
    ]

    PLACE_FIELDS = [
        "contained_within",  # Returns the identifiers of known places that contain the referenced place.
        "country_code",
        "country",  # full-length name of country this place belongs to
        "full_name",  # long-form
        "geo",  # Contains GeoJSON (bbox, type, properties)
        "id",  # used to programmatically retrieve place
        "name",  # Short name
        "place_type",  # city, etc.
    ]

    def __init__(self) -> None:
        load_dotenv()
        self.api = tweepy.Client(os.environ.get("BEARER_TOKEN"))

    def parse_twitter_response(self, twitter_respose):
        tweets_dict = {}
        places = {}

        # No tweets returned
        if not twitter_respose.data:
            return tweets_dict

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

    def search_all_tweets(self, **args):
        tweets = {}
        try:
            for response in tweepy.Paginator(
                self.api.search_all_tweets,
                tweet_fields=self.TWEET_FIELDS,
                place_fields=self.PLACE_FIELDS,
                user_fields=self.USER_FIELDS,
                expansions=["author_id", "geo.place_id", "attachments.media_keys"],
                **args
            ):
                # To mitigate "Too Many Requests" twitter API error
                sleep(1)
                tweets.update(self.parse_twitter_response(response))
        except TooManyRequests:
            print("Too many requests, returning currently extracted tweets")

        return tweets

    def get_tweets_from_id(self, ids: list[int]):
        tweets = {}
        #  Twitter API accepts 100 tweets per request, so they are seperated
        #  in a list of lists that have 100 tweet ids at most
        list_of_list_of_ids = np.array_split(ids, math.ceil(len(ids) / 100))

        for list_of_ids in list_of_list_of_ids:
            twitter_respose = self.api.get_tweets(
                ids=",".join(map(str, list_of_ids)),
                expansions=["author_id", "geo.place_id", "attachments.media_keys"],
                place_fields=self.PLACE_FIELDS,
                tweet_fields=self.TWEET_FIELDS,
                user_fields=self.USER_FIELDS,
            )

            tweets.update(self.parse_twitter_response(twitter_respose))

        return tweets
