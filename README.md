### Description

The project focuses on extracting context related to flooding events in Sweden and present them
using spatio-temporal plots.

### Setting up the environment

**Install python packages** with `pip` or any virtual environment using `requirements.txt`

**Install corpora** from spacy and nltk for NLP tasks `make install_corpora`

To **get the data**, execute `dvc pull`. A web page will open that will require you
to get permission for the directory containing the data. You will have to wait
until to give you permission to complete this step.

### Usage

To train the flood classifier `make train_flood_classifier`

The pipeline runs on AWS Sagemaker by default, to run it locally, use the
following:

`dvc exp run train --set-param 'datasets=${supervisor.processed}' --set-param 'env=${envs.locally}'`

To show the metric for experiments: `make evaluation`

To be able to leverage twitter's API, create `.env` to store the credentials with a similar format of `.env.template`.
