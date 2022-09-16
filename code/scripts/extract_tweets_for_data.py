#!/usr/bin/env python3
from TwitterAdapter import TwitterAdapter
import numpy as np
import math
import json
import csv
from deep_translator import GoogleTranslator
import sys

"""
 Utility script to extract twitter for available data (annotated_tweets.csv)
"""

def main() -> None:
    twitter = TwitterAdapter()

    tweets = {}
    tweets_extra_data = {}

    with open('./data/annotated_tweets.csv', newline='') as csvfile:

        rows = list(csv.DictReader(csvfile, delimiter=','))
        tweets_extra_data = dict((row['id'], row) for row in rows)

        #  Twitter API accepts 100 tweets per request, so they are seperated
        #  in a list of lists that have 100 tweet ids at most
        list_of_rows = np.array_split(rows, math.ceil(len(rows)/100))

        for part_of_rows in list_of_rows:
            ids = [row['id'] for row in part_of_rows]
            twitter_respose = twitter.api.get_tweets(ids=ids,
                    tweet_fields =twitter.TWEET_FIELDS_EXTRACTED, expansions="geo.place_id",
                    place_fields=twitter.PLACE_FIELDS)

            tweets.update(twitter.parse_twitter_response(twitter_respose))

    for tweet_id in tweets.keys():
        tweets[tweet_id].update(tweets_extra_data[str(tweet_id)])
        tweets[tweet_id]['text_en'] = GoogleTranslator(source='auto', target='en').translate(tweets[tweet_id]['text'])

    # Write pretty print JSON data to file
    with open(f"{twitter.OUTPUT_PATH}/annotated_tweets_extracted.json", "w") as write_file:
        json.dump(tweets, write_file, indent=4)


if __name__ == '__main__':
    sys.exit(main())

