#!/usr/bin/env python3
import sys
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from datasets.filesystems.s3filesystem import S3FileSystem
from datasets.arrow_dataset import Dataset
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from datasets.dataset_dict import DatasetDict
from transformers.trainer import Trainer
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
from transformers.training_args import TrainingArguments

"""
Training dataset/s using either sagemaker or locally

args:
    1- method: locally, sagemaker
    2- target label: relevant, mentions_impact
    *- dataset: path to dataset
"""

model_name: str = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
target_label = ""


def train_locally(dataset) -> None:
    batch_size = 64
    logging_steps = len(dataset["train"]) // batch_size
    output_model_name = "bert-base-uncased-finetuned-floods"
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of labels for classification task
    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_model_name,
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

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
    )

    trainer.train()
    preds_output = trainer.predict(dataset["test"])
    print(preds_output.metrics)


def train_sagemaker(dataset):
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
        role = iam.get_role(RoleName="sage_maker")["Role"]["Arn"]
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    # Storing data in s3 bucket
    s3 = S3FileSystem()
    s3_prefix = "samples/datasets/floods"

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    save_to_s3 = False
    training_input_path = f"s3://{sess.default_bucket()}/{s3_prefix}/train"
    test_input_path = f"s3://{sess.default_bucket()}/{s3_prefix}/test"

    if save_to_s3:
        # save train_dataset to s3
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        # train_dataset.save_to_disk(training_input_path, fs=s3)

        # save test_dataset to s3
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        test_dataset.save_to_disk(test_input_path, fs=s3)

    hyperparameters = {"epochs": 1, "train_batch_size": 32, "model_name": model_name}

    huggingface_estimator = HuggingFace(
        entry_point="sagemaker_train.py",
        source_dir="src/train/scripts",
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        role=role,
        transformers_version="4.12",
        pytorch_version="1.9",
        py_version="py38",
        hyperparameters=hyperparameters,
    )

    huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})

    from sagemaker.s3 import S3Downloader

    print(huggingface_estimator.model_data)
    # Get directory of output for model
    eval_s3_URI = (
        "/".join(huggingface_estimator.model_data.split("/")[:-1]) + "/output.tar.gz"
    )
    S3Downloader.download(
        # s3_uri=huggingface_estimator.model_data, # S3 URI where the trained model is located
        s3_uri=eval_s3_URI,  # S3 URI where the trained model is located
        local_path=".",  # local path where *.targ.gz is saved
        sagemaker_session=sess,  # SageMaker session used for training the model
    )
    print("Model metrics can be found output.tar.gz")


def get_datasets(list_path) -> DatasetDict:
    df_last = pd.DataFrame()

    # Use datasets that has the target label only
    for path in list_path:
        df = pd.read_csv(path)
        if target_label not in df.columns:
            raise Exception(f"{target_label} label doesn't exist in {path} dataset")

        df = df.rename(columns={target_label: "label"})
        df = df[["id", "text", "label"]]
        df_last = pd.concat([df_last, df])

    dataset = Dataset.from_pandas(df)

    train_testvalid = dataset.train_test_split(test_size=0.1)

    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    train_test_valid_dataset = DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    train_test_valid_dataset_tokenized = train_test_valid_dataset.map(
        tokenize, batched=True, batch_size=None
    )

    return train_test_valid_dataset_tokenized


def main():
    global target_label
    method = sys.argv[1]
    target_label = sys.argv[2]
    print(target_label)
    available_labels = ["mentions_impact", "relevant"]
    if target_label not in available_labels:
        raise Exception(
            f"{target_label} is not valid, it should be one of the"
            f" following {', '.join(available_labels)}"
        )
    dataset = get_datasets(sys.argv[3:])
    if method == "sagemaker":
        train_sagemaker(dataset)
    elif method == "locally":
        train_locally(dataset)
    else:
        raise Exception(f"{method} is not valid method")


if __name__ == "__main__":
    main()
