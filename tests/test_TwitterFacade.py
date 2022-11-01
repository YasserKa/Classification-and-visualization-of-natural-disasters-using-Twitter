import datetime

from tweepy import Media, Place, Response, Tweet, User

from flood_detection.classes.twitter_facade import TwitterFacade


# TODO: add test for attachements
class TestTwitterFacade:
    def test_parse_twitter_response(self) -> None:
        tweet_data = {
            "id": "1",
            "text": "This is a tweet",
            "author_id": "1",
            "geo": {"place_id": "1"},
            "lang": "sv",
            "edit_history_tweet_ids": ["2"],
        }
        place = {"name": "place1", "id": "1", "full_name": "full_name1"}
        user = {"id": "1", "username": "username", "name": "name"}
        response = Response(
            data=[Tweet(tweet_data)],
            includes={"places": [Place(place)], "users": [User(user)]},
            meta=None,
            errors=None,
        )

        parsed_tweet = {
            1: {
                "id": "1",
                "text": "This is a tweet",
                "lang": "sv",
                "edit_history_tweet_ids": ["2"],
                "user": {"id": "1", "username": "username", "name": "name"},
                "place": {"name": "place1", "id": "1", "full_name": "full_name1"},
            }
        }
        twitter = TwitterFacade()
        result = twitter.parse_twitter_response(response)
        assert result == parsed_tweet

    def test_extract_from_ids(self):
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
                "lang": "sv",
                "text": "RT @chrissieSTH: Lyssnar på nyheterna.. Katastrof, katastrof och allt är extremt. Men i morse, mitt i allt extremt så säger meteorologen Ma…",
                "user": {
                    "protected": False,
                    "location": "Sweden",
                    "name": "POGO",
                    "pinned_tweet_id": "1583260532773388289",
                    "id": "903741974456520705",
                    "username": "Pogo_Pedagog2",
                    "verified": False,
                    "description": "Meme war veteran.\nTwitter-winter survivor.",
                    "profile_image_url": "https://pbs.twimg.com/profile_images/1586084217695379456/efr1pG73_normal.jpg",
                    "created_at": "2017-09-01T22:10:52.000Z",
                },
            }
        }
        twitter = TwitterFacade()
        tweets = twitter.get_tweets_from_ids(id)
        assert tweets == result
