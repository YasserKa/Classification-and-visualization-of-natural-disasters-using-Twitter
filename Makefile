install: setup_env install_corpora

setup_env:
	poetry install
	poetry run pre-commit install

test:
	python -m pytest

install_corpora: 
	python -m spacy download en_core_web_sm
	python -m spacy download sv_core_news_sm
	python -m nltk.downloader omw-1.4 -d ./nltk_data
	python -m nltk.downloader wordnet -d ./nltk_data

pull_data:
	python -m dvc pull

train_flood_classifier:
	dvc exp run train --set-param 'datasets=${supervisor.processed}' -f

evaluation:
	dvc exp show --drop '.*' --keep "(Experiment|Created|eval_accuracy|eval_f1|eval_precision|eval_recall)" -n 2

pipeline_for_supervisor:
	python ./flood_detection/data/preprocess.py "./data/tweets/supervisor_annotated_tweets.json"
	python ./flood_detection/predict/predict_floods.py --dataset_path "./data/processed/supervisor_annotated_tweets.csv"
	python ./flood_detection/predict/extract_location.py "./data/processed_flood/supervisor_annotated_tweets.csv" 
	python ./flood_detection/visualize/dash_app.py "./data/processed_geo/supervisor_annotated_tweets.csv" 

pipline_from_api:
	python ./flood_detection/data/extract_tweets_from_api.py --from $(from) --to $(to)
	python ./flood_detection/data/preprocess.py "data/tweets/twitter_api_$(from)_to_$(to).json"
	python ./flood_detection/predict/predict_floods.py --dataset_path "data/processed/twitter_api_$(from)_to_$(to).csv"
	python ./flood_detection/predict/extract_location.py "data/processed_flood/twitter_api_$(from)_to_$(to).csv" 
	python ./flood_detection/visualize/dash_app.py "data/processed_geo/twitter_api_$(from)_to_$(to).csv" 
