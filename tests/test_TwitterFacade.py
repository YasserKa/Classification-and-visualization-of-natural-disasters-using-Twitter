from tweepy import Place, Response, Tweet

from src.classes.twitter_facade import TwitterFacade


class TestTwitterFacade:
    def test_parse_twitter_response(self) -> None:
        tweet_data = {
            "id": "1",
            "text": "This is a tweet",
            "author_id": "2",
            "geo": {"place_id": "1"},
            "lang": "sv",
        }
        place = {"name": "place1", "id": "1", "full_name": "full_name1"}
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
                "place": {"name": "place1", "id": "1", "full_name": "full_name1"},
            }
        }
        twitter = TwitterFacade()
        result = twitter.parse_twitter_response(response)
        assert result == parsed_tweet
