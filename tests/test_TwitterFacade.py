import datetime

from tweepy import Place, Response, Tweet

from flood_detection.classes.twitter_facade import TwitterFacade


class TestTwitterFacade:
    def test_parse_twitter_response(self) -> None:
        tweet_data = {
            "id": "1",
            "text": "This is a tweet",
            "author_id": "2",
            "geo": {"place_id": "1"},
            "lang": "sv",
            "edit_history_tweet_ids": ["2"],
        }
        place = {"name": "place1", "id": "1", "full_name": "full_name1"}
        print()
        response = Response(
            data=[Tweet(tweet_data)],
            includes={"places": [Place(place)]},
            meta=None,
            errors=None,
        )

        parsed_tweet = {
            1: {
                "id": "1",
                "text": "This is a tweet",
                "author_id": "2",
                "lang": "sv",
                "edit_history_tweet_ids": ["2"],
                "place": {"name": "place1", "id": "1", "full_name": "full_name1"},
            }
        }
        twitter = TwitterFacade()
        result = twitter.parse_twitter_response(response)
        assert result == parsed_tweet

    def test_extract_from_id(self):
        id = [1428506193014796289]
        result = {
            1428506193014796289: {
                "conversation_id": "1428506193014796289",
                "public_metrics": {
                    "retweet_count": 10,
                    "reply_count": 0,
                    "like_count": 0,
                    "quote_count": 0,
                },
                "id": "1428506193014796289",
                "created_at": "2021-08-19T23:56:31.000Z",
                "source": "Twitter for Android",
                "edit_controls": {
                    "edits_remaining": 5,
                    "is_edit_eligible": True,
                    "editable_until": datetime.datetime(
                        2021, 8, 20, 0, 26, 31, tzinfo=datetime.timezone.utc
                    ),
                },
                "referenced_tweets": [
                    {"type": "retweeted", "id": "1428450964428709892"}
                ],
                "edit_history_tweet_ids": ["1428506193014796289"],
                "reply_settings": "everyone",
                "entities": {
                    "mentions": [
                        {
                            "start": 3,
                            "end": 15,
                            "username": "chrissieSTH",
                            "id": "83788155",
                        }
                    ]
                },
                "author_id": "903741974456520705",
                "lang": "sv",
                "text": "RT @chrissieSTH: Lyssnar på nyheterna.. Katastrof, katastrof och allt är extremt. Men i morse, mitt i allt extremt så säger meteorologen Ma…",
            }
        }
        twitter = TwitterFacade()
        tweets = twitter.get_tweets_from_id(id)
        assert tweets == result
