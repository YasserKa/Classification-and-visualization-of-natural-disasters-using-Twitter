#!/usr/bin/env python

import os

import click
import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


@click.command()
@click.option(
    "--dataset_path",
    required=True,
    help="Path to dataset to do the predictions on",
)
def main(dataset_path):

    with initialize(version_base=None, config_path="../../conf"):
        cfg: DictConfig = compose(config_name="config")
    path_to_model = cfg.models_dir.flood_detection

    dataset_path = abspath(dataset_path)

    if dataset_path == abspath(cfg.supervisor.processed):
        output_path: str = cfg.supervisor.processed_flood
    elif dataset_path.startswith(abspath(cfg.twitter_api.processed)):
        output_file_name = os.path.basename(dataset_path)
        output_path: str = abspath(
            "./"
            + os.path.dirname(cfg.twitter_api.processed_flood)
            + "/"
            + output_file_name
        )
    else:
        raise Exception(f"{dataset_path} file not found")

    df = pd.read_csv(dataset_path)

    dataset: Dataset = Dataset.from_pandas(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBertForSequenceClassification.from_pretrained(
        f"{path_to_model}/pytorch_model.bin",
        config=f"{path_to_model}/config.json",
        local_files_only=True,
    ).to(device)

    tokenizer = DistilBertTokenizer.from_pretrained(
        f"{path_to_model}/vocab.txt", local_files_only=True
    )

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    def forward_pass_with_label(batch):
        # Place all input tensors on the same device as the model
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in tokenizer.model_input_names
        }
        with torch.no_grad():
            output = model(**inputs)
            pred_label = torch.argmax(output.logits, axis=-1)
            return {
                "predicted_label": pred_label.cpu().numpy(),
            }

    tokenized = dataset.map(tokenize, batched=True, batch_size=None)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    tokenized = tokenized.map(forward_pass_with_label, batched=True, batch_size=16)
    tokenized.set_format("pandas")
    df_output = tokenized[:][:]
    df_output = df_output.drop(columns=["input_ids", "attention_mask"])

    df_output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
