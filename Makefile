install: setup_env install_corpora

setup_env:
	poetry install
	poetry run pre-commit install

test:
	python -m pytest

install_corpora: 
	python -m spacy download en_core_web_sm
	python -m nltk.downloader omw-1.4 -d ./nltk_data
	python -m nltk.downloader wordnet -d ./nltk_data

pull_data:
	python -m dvc pull

train_flood_classifier:
	dvc exp run train --set-param 'datasets=${supervisor.processed}' -f

evaluation:
	dvc exp show --drop '.*' --keep "(Experiment|Created|eval_accuracy|eval_f1|eval_precision|eval_recall)" -n 2

extract_from_api:
	python ./flood_detection/data/extract_tweets_from_api.py --from $(from) --to $(to)
	python ./flood_detection/data/preprocess.py data/tweets/twitter_api.json 
	python ./flood_detection/predict/predict_floods.py --dataset_path data/processed/twitter_api.csv 
	python ./flood_detection/predict/extract_location.py data/processed_flood/twitter_api.csv 
	python ./flood_detection/visualize/dash_app.py ./data/processed_geo/twitter_api.csv 
