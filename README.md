### Description

The project focuses on extracting context related to flooding events from tweets
using Natural Language Processing, mainly in Sweden.

The file structure of the project is inspired by [khuyentran1401/data-science-template: Template for a data science project](https://github.com/khuyentran1401/data-science-template)

### Setting up the environment

**Install python packages** with `pip` or any virtual environment using `requirements.txt`

Tools used:

- [DVC](https://dvc.org/) for managing data and pipelines
- [Plotly](https://plotly.com/dash/) for visualization

Install spacy's pipeline package using `python -m spacy download en_core_web_sm`

To **get the data**, execute `dvc pull`. A web page will open that will require you
to get permission for the directory containing the data. You will have to wait
until to give you permission to complete this step.

### Usage

**To train the flood classifier**
`dvc exp run train --set-param 'datasets=${supervisor.processed}'`

The pipeline runs on AWS Sagemaker by default, to run it locally, use the
following
`dvc exp run train --set-param 'datasets=${supervisor.processed}' --set-param 'env=${envs.locally}'`

**Show metric for experiments**
`dvc exp show --drop '.*' --keep "(Experiment|Created|eval_accuracy|eval_f1|eval_precision|eval_recall)" -n 2`

### Training the model

_Extra_:
To be able to leverage twitter's API, use `.env.template` to create `.env`.
