#!/usr/bin/env python3
from dotenv import load_dotenv
import tweepy
import os 

class TwitterAdapter(object):
    """
    Singletong class that uses tweepy to extract tweets
    """

    _instance = None

    OUTPUT_PATH = "./output"

    # Tweets fields can be found here https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
    TWEET_FIELDS_AVAILABLE = ["attachments", "author_id", "context_annotations",
            "conversation_id", "created_at", "entities", "geo", "id",
            "in_reply_to_user_id", "lang", "non_public_metrics", "organic_metrics",
            "possibly_sensitive", "promoted_metrics", "public_metrics",
            "referenced_tweets", "reply_settings", "source", "text", "withheld"]

    TWEET_FIELDS_EXTRACTED = ["id", "text", "public_metrics", "referenced_tweets",
            "author_id", "attachments", "geo", "source", "lang", "created_at"]

    PLACE_FIELDS = ["contained_within", "country", "country_code", "full_name",
            "geo", "id", "name", "place_type"]


    # Making it a singelton class
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TwitterAdapter, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def __init__(self):
        load_dotenv()
        bearer_token = os.environ.get('BEARER_TOKEN')
        self.api = tweepy.Client(bearer_token)


    def parse_twitter_response(self, twitter_respose):
        tweets_dict = {}
        places = {}

        if 'places' in twitter_respose.includes:
            for place in twitter_respose.includes['places']:
                places[place.data['id']] = place.data

        for tweet in twitter_respose.data:
            tweets_dict[tweet['id']] = tweet.data
            if 'geo' in tweet:
                tweets_dict[tweet['id']]['place'] = places[tweet['geo']['place_id']]
                del tweets_dict[tweet['id']]['geo']
            tweets_dict[tweet['id']]['tweet_url'] = f"https://twitter.com/{tweet['author_id']}/status/{tweet['id']}"

        return tweets_dict
