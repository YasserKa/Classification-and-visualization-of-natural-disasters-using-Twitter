extract_tweets: code/scripts/extract_tweets.py
	python -m code.scripts.extract_tweets

extract_tweets_for_data: code/scripts/extract_tweets_for_data.py
	python -m code.scripts.extract_tweets_for_data

do_tests: code/tests
	python -m pytest code/tests/
