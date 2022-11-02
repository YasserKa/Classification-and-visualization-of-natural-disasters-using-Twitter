#!/usr/bin/env python3

import json
import tarfile
from enum import Enum

import boto3
import click
import pandas as pd
import sagemaker
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.filesystems.s3filesystem import S3FileSystem
from hydra import compose, initialize
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig
from sagemaker.huggingface import HuggingFace
from sagemaker.s3 import S3Downloader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

"""
Training dataset/s locally or using sagemaker
"""

# REFACTOR: Remove train_locally() and use train.py instead


class Environment(Enum):
    SAGEMAKER = "sagemaker"
    LOCALLY = "locally"


class Label(Enum):
    RELEVANT = "relevant"
    MENTIONS_IMPACT = "mentions_impact"


class ModelEncapsualtion:
    """Model used to train the dataset"""

    def __init__(self, name, labels_num=2):
        self.__name = name
        self.__tokenizer = AutoTokenizer.from_pretrained(name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=labels_num
        ).to(device)

    def get_model(self):
        return self.__model

    def get_name(self):
        return self.__name

    def get_tokenizer(self):
        return self.__tokenizer

    def tokenize(self, batch):
        return self.__tokenizer(batch["text"], padding="max_length", truncation=True)


def train_locally(dataset, model_enc, output_dir, download_model) -> None:
    batch_size = 4
    logging_steps = len(dataset["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True,
        log_level="error",
    )

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    trainer = Trainer(
        model=model_enc.get_model(),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=model_enc.get_tokenizer(),
    )

    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=dataset["test"])

    if download_model:
        trainer.save_model(output_dir)
    with open("eval.json", "w") as file:
        json.dump(eval_result, file)


def train_sagemaker(dataset, model_enc, role_name, download_model):
    # Getting session/role
    sess = sagemaker.Session()
    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it not exists
    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    # Storing data in s3 bucket
    s3 = S3FileSystem()
    s3_prefix = "samples/datasets/floods"

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    save_to_s3 = True
    training_input_path = f"s3://{sess.default_bucket()}/{s3_prefix}/train"
    test_input_path = f"s3://{sess.default_bucket()}/{s3_prefix}/test"

    if save_to_s3:
        # save train_dataset to s3
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        train_dataset.save_to_disk(training_input_path, fs=s3)

        # save test_dataset to s3
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        test_dataset.save_to_disk(test_input_path, fs=s3)

    hyperparameters = {
        "epochs": 1,
        "train_batch_size": 32,
        "model_name": model_enc.get_name(),
    }

    huggingface_estimator = HuggingFace(
        entry_point="sagemaker_train.py",
        source_dir="flood_detection/train/scripts",
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        role=role,
        transformers_version="4.12",
        pytorch_version="1.9",
        py_version="py38",
        hyperparameters=hyperparameters,
    )

    huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})

    # Get directory of output for model
    output = "output.tar.gz"
    eval_s3_URI = (
        "/".join(huggingface_estimator.model_data.split("/")[:-1]) + f"/{output}"
    )
    # Download evaluation
    S3Downloader.download(
        s3_uri=eval_s3_URI,  # S3 URI where the trained model is located
        local_path=".",  # local path where *.targ.gz is saved
        sagemaker_session=sess,  # SageMaker session used for training the model
    )
    with tarfile.open(output) as tar:
        tar.extractall()

    if download_model:
        S3Downloader.download(
            s3_uri=huggingface_estimator.model_data,  # S3 URI where the trained model is located
            local_path=".",  # local path where *.targ.gz is saved
            sagemaker_session=sess,  # SageMaker session used for training the model
        )


def get_tweets_from_paths(path_list, label_enum):
    """
    Concatenate datasets from list of paths to datasets
    """
    df_last = pd.DataFrame()

    # Use datasets that has the target label only
    for path in path_list:
        df = pd.read_csv(path)
        if label_enum.value not in df.columns:
            raise Exception(f"{label_enum.value} label doesn't exist in {path} dataset")

        df = df.rename(columns={label_enum.value: "label"})

        df = df[["id", "text", "label"]]
        df_last = pd.concat([df_last, df])

    return df_last


def get_dataset(
    path_list: list[str], model_enc: ModelEncapsualtion, label_enum: Label
) -> DatasetDict:
    """
    Get a train/valid/test splitted tokenized dataset from a list of datasets
    """
    df: pd.DataFrame = get_tweets_from_paths(path_list, label_enum)

    # Undersample tweets that don't mention impact
    if label_enum == Label.MENTIONS_IMPACT:
        ros: RandomUnderSampler = RandomUnderSampler(random_state=0)
        X_labels: list[str] = list(df.columns != "label")
        y_label: str = "label"
        df_undersampled: pd.DataFrame = pd.DataFrame()
        X, y = ros.fit_resample(df.loc[:, X_labels], df[y_label])
        df_undersampled: pd.DataFrame = X
        df_undersampled[y_label] = y
        # Shuffle
        df: pd.DataFrame = df_undersampled.sample(frac=1)

    dataset: Dataset = Dataset.from_pandas(df)

    train_testvalid = dataset.train_test_split(test_size=0.1)

    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    train_test_valid_dataset = DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    train_test_valid_dataset_tokenized = train_test_valid_dataset.map(
        model_enc.tokenize, batched=True, batch_size=None
    )

    return train_test_valid_dataset_tokenized


@click.command()
@click.option(
    "--environment", "--env", "env", default="sagemaker", help="Training environment"
)
@click.option("--model", default="distilbert-base-uncased", help="Model used to train")
@click.option("--label", default="relevant", help="Target label to train the model for")
@click.option(
    "--role_name", default="sage_maker", help="AWS role_name to access resources"
)
@click.option(
    "--download_model", is_flag=True, default=False, help="Download trained model"
)
@click.argument("datasets_path", nargs=-1)
def main(env, model, label, datasets_path, role_name, download_model):
    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")

    output_dir: str = f"{cfg.models_dir}/flood_detection"

    model_enc = ModelEncapsualtion(model)
    label_enum = Label(label)
    dataset = get_dataset(datasets_path, model_enc, label_enum)
    env_enum = Environment(env)

    if env_enum == Environment.SAGEMAKER:
        train_sagemaker(dataset, model_enc, role_name, download_model)
    elif env_enum == Environment.LOCALLY:
        train_locally(dataset, model_enc, output_dir, download_model)
    else:
        raise Exception(f"{env} is not valid environment")


if __name__ == "__main__":
    main()
