#!/bin/env python3
from dotenv import load_dotenv
from datetime import datetime
import tweepy
import json
import os 

load_dotenv()

consumer_key = os.environ.get('CONSUMER_KEY')
consumer_secret = os.environ.get('CONSUMER_SECRET')
bearer_token = os.environ.get('BEARER_TOKEN')

api = tweepy.Client(bearer_token)

tweet_fields_available = ["attachments", "author_id", "context_annotations",
        "conversation_id", "created_at", "entities", "geo", "id",
        "in_reply_to_user_id", "lang", "non_public_metrics", "organic_metrics",
        "possibly_sensitive", "promoted_metrics", "public_metrics",
        "referenced_tweets", "reply_settings", "source", "text", "withheld"]

tweet_fields_extracted = ["id", "text", "public_metrics", "referenced_tweets",
        "author_id", "entities", "context_annotations", "attachments", "geo", "source"]


def print_tweets(tweets):
    for tweet in tweets:
        print(tweet['text'])

tweets_response = api.search_all_tweets(query="(floods OR flood OR"
        " översvämning OR översvämningar) place_country:SE has:geo", 
        tweet_fields=tweet_fields_extracted, expansions="geo.place_id",
        start_time= datetime(2021,8,17),
        end_time= datetime(2021,8,19),
        place_fields='contained_within,country,country_code,full_name,geo,id,name,place_type', max_results=10)
 

tweets_dict = {}

# NOTE: includes has 'places' : array of place, can be extracted using
# tweets_response.includes.place.data
for tweet in tweets_response.data:
    tweets_dict[tweet['id']] = tweet.data
    tweets_dict[tweet['id']]['tweet_url'] = f"https://twitter.com/{tweet['author_id']}/status/{tweet['id']}"


# Write pretty print JSON data to file
with open("tweets.json", "w") as write_file:
    json.dump(tweets_dict, write_file, indent=4)
