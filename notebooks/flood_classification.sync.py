# %%
import json

import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub import notebook_login
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from src.data.text_processing import TextProcessing

# %% [markdown]
# ## Preprocessing data
#
# Loading available datasets and  transforming it to pandas dataframe with columns
# `text` and `label`, where the label shows if the text is on-topic (1) or
# off-topic (0)

# %% [markdown]
# Data obtained from the supervisor

# %%
with initialize(version_base=None, config_path="../conf"):
    cfg: DictConfig = compose(config_name="config")
    supervisor_tweets_path = abspath("../" + cfg.supervisor.tweets)
    alberta_path = abspath("../" + cfg.alberta.raw)
    queensland_path = abspath("../" + cfg.queensland.raw)

with open(supervisor_tweets_path, "r") as file:
    tweets_json = json.load(file)
supervisor_df = pd.json_normalize(list(tweets_json.values()))
print(supervisor_df.shape)
# NOTE: Some tweets, don't have geo-tagging or attachements, hence the NaN values
supervisor_df.head()

# %%
supervisor_df = supervisor_df[["id", "text_en", "On Topic"]]
supervisor_df = supervisor_df.rename(columns={"text_en": "text", "On Topic": "label"})
supervisor_df = supervisor_df[supervisor_df["label"] != ""]
supervisor_df = supervisor_df.astype({"label": "int", "id": "int"})

# %% [markdown]
# Data obtaine from [CrisisLex: Download Crisis-Related Collections](https://crisislex.org/data-collections.html#CrisisLexT6)

# %%
queensland_df = pd.read_csv(queensland_path)
alberta_df = pd.read_csv(alberta_path)

print(f"queensland {queensland_df.shape}")
print(f"alberta {alberta_df.shape}")
# NOTE: Some tweets, don't have geo-tagging or attachements, hence the NaN values
queensland_df.head()


# %%
def process_crisislex_data(df):
    df = df.rename(columns={"tweet_id": "id", "tweet": "text"})
    df["label"] = df["label"].apply(lambda x: 1 if x == "on-topic" else 0)
    df["id"] = df["id"].apply(lambda x: x[1:-1])
    df = df.astype({"id": "int"})
    return df


queensland_df = process_crisislex_data(queensland_df)
alberta_df = process_crisislex_data(alberta_df)

# %%
df_needed = pd.concat([queensland_df, alberta_df, supervisor_df])

# %%
# Remove retweets
df_needed = df_needed[df_needed["text"].str[:2] != "RT"]
# Remove duplicates
df_needed = df_needed.drop_duplicates()

text_preprocessing = TextProcessing()
# Clean text
df_needed["text"] = text_preprocessing.clean_text(df_needed["text"].to_numpy())

# %%
df_needed.head()

# %%
dataset = Dataset.from_pandas(df_needed)

dataset = dataset.remove_columns(["id"])

train_testvalid = dataset.train_test_split(test_size=0.1)

test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

train_test_valid_dataset = DatasetDict(
    {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"],
    }
)

# %% [markdown]
# ## Finetuning the model
#
# Use hugging face library

# %%
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)


# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)


train_test_valid_dataset_tokenized = train_test_valid_dataset.map(
    tokenize, batched=True, batch_size=None
)

# %%
notebook_login()


# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


batch_size = 64
logging_steps = len(train_test_valid_dataset_tokenized["train"]) // batch_size
model_name = "bert-base-uncased-finetuned-floods"
torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir=model_name,
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

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_test_valid_dataset_tokenized["train"],
    eval_dataset=train_test_valid_dataset_tokenized["valid"],
    tokenizer=tokenizer,
)

trainer.train()
