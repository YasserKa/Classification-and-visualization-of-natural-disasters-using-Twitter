FROM python:3.10-slim
COPY ./docker/requirements.txt .
RUN pip install -r requirements.txt
COPY ./setup.py .
COPY ./flood_detection ./flood_detection
RUN pip install .
RUN	python -m spacy download en_core_web_sm
RUN	python -m spacy download sv_core_news_sm
RUN	python -m nltk.downloader omw-1.4 -d ./nltk_data
RUN	python -m nltk.downloader wordnet -d ./nltk_data
COPY ./docker/data ./data
CMD python ./flood_detection/visualize/dash_app.py ./data/processed_geo/twitter_api_2021-08-17_to_2021-08-23__2023-02-02_22:34:09.csv
