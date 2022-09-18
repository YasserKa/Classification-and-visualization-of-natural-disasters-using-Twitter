extract_tweets: flood_detection/scripts/extract_tweets.py
	python -m flood_detection.scripts.extract_tweets

extract_tweets_for_data: flood_detection/scripts/extract_tweets_for_data.py
	python -m flood_detection.scripts.extract_tweets_for_data

do_tests: flood_detection/tests
	python -m pytest flood_detection/tests/
