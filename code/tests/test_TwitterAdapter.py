from TwitterAdapter import TwitterAdapter
from tweepy import Response, Place, Tweet


class TestTwitterAdapter:
    def test_parse_twitter_response(self):
        response = Response(
                data= [Tweet({ "id":"1", "text":'This is a tweet', "author_id":"2", "geo": {'place_id': "1"}, "lang": 'sv'})]
                , includes={'places':[Place({ 'name': 'place1', 'id': "1",
                    'full_name': 'full_name1'})] },
                meta=None,
                errors=None)

        parsed_tweet = {1: {
            "id":"1",
            "text":'This is a tweet',
            "author_id": "2",
            "lang": 'sv',
            'place': { 'name': 'place1', 'id': "1", 'full_name': 'full_name1'},
            'tweet_url':"https://twitter.com/2/status/1"
            }}
        twitter = TwitterAdapter()
        result = twitter.parse_twitter_response(response)
        assert result == parsed_tweet

