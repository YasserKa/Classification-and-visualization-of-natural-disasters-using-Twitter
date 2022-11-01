.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install_spacy_packages: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
	python -m spacy download en_core_web_sm

pull_data:
	python -m dvc pull

test:
	python -m pytest

extract_from_api:
	python ./flood_detection/data/extract_tweets_from_api.py --from $(from) --to $(to)
	python ./flood_detection/data/preprocess.py data/tweets/twitter_api.json 
	python ./flood_detection/predict/predict_floods.py --dataset_path data/processed/twitter_api.csv 
	python ./flood_detection/predict/extract_location.py data/processed_flood/twitter_api.csv 
	python ./flood_detection/visualize/dash_app.py ./data/processed_geo/twitter_api.csv 
