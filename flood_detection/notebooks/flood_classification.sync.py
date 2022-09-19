# %%
import pandas as pd
import json
from flood_detection.twitter import utils

# %% [markdown]
# ## Preprocessing data
#
# Loading available datasets and  transforming it to pandas dataframe with columns
# `text` and `label`, where the label shows if the text is on-topic (1) or
# off-topic (0)

# %% [markdown]
# Data obtained from the supervisor

# %%
with open(f"../output/annotated_tweets_extracted.json", "r") as file:
    tweets_json = json.load(file)
supervisor_df = pd.json_normalize(list(tweets_json.values()))
print(supervisor_df.shape)
# NOTE: Some tweets, don't have geo-tagging or attachements, hence the NaN values
supervisor_df.head()

# %%
supervisor_df = supervisor_df[['id', 'text_en', 'On Topic']]
supervisor_df = supervisor_df.rename(columns={"text_en": "text", "On Topic": "label"})
supervisor_df = supervisor_df[supervisor_df['label'] != '']
supervisor_df = supervisor_df.astype({'label': 'int', 'id': 'int'})

# %% [markdown]
# Data obtaine from [CrisisLex: Download Crisis-Related Collections](https://crisislex.org/data-collections.html#CrisisLexT6)

# %%
queensland_df = pd.read_csv("../data/CrisisLexT6/2013_Queensland_Floods/2013_Queensland_Floods-ontopic_offtopic.csv")
alberta_df = pd.read_csv("../data/CrisisLexT6/2013_Alberta_Floods/2013_Alberta_Floods-ontopic_offtopic.csv")

print(f"queensland {queensland_df.shape}")
print(f"alberta {alberta_df.shape}")
# NOTE: Some tweets, don't have geo-tagging or attachements, hence the NaN values
queensland_df.head()


# %%
def process_crisislex_data(df):
    df = df.rename(columns={"tweet_id": "id", "tweet": "text"})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'on-topic' else 0)
    # Remove single quotation marks (') from start and end
    df['id'] = df['id'].apply(lambda x: x[1:-1])
    df = df.astype({'id': 'int'})
    return df

queensland_df = process_crisislex_data(queensland_df)
alberta_df = process_crisislex_data(alberta_df)

# %%
df_needed = pd.concat([queensland_df, alberta_df, supervisor_df])

# %%
# Remove retweets
df_needed = df_needed[df_needed['text'].str[:2] != "RT"]
# Remove duplicates
df_needed = df_needed.drop_duplicates()
# Clean text
df_needed['text'] = df_needed['text'].apply(lambda x: utils.clean_text(x))

# %%
df_needed.head()

# %%
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df_needed)

dataset = dataset.remove_columns(['id'])

train_testvalid = dataset.train_test_split(test_size=0.1)

test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# %% [markdown]
# ## Finetuning the model
#
# Use hugging face library

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2

model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels).to(device))

# tokenizer helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

train_test_valid_dataset_tokenized = train_test_valid_dataset.map(tokenize, batched=True, batch_size=None)

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(train_test_valid_dataset_tokenized["train"]) // batch_size
model_name = f"bert-base-uncased-finetuned-floods"
torch.cuda.empty_cache()

training_args = TrainingArguments(output_dir=model_name,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True, 
        log_level="error")

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_test_valid_dataset_tokenized["train"],
                  eval_dataset=train_test_valid_dataset_tokenized["valid"],
                  tokenizer=tokenizer)

trainer.train()

# %% [markdown]
#
# TODO: Try too use sagemaker to fine tune the transformer

# %%
import sagemaker
import sagemaker.huggingface
import boto3

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sage_maker')['Role']['Arn']
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

# %%
print(train_test_valid_dataset_tokenized['train'])

# %%
import botocore
import datasets

s3 = datasets.filesystems.S3FileSystem() 
s3_prefix = 'samples/datasets/floods'

train_dataset = train_test_valid_dataset_tokenized['train']
test_dataset = train_test_valid_dataset_tokenized['test']

train_dataset = train_dataset.remove_columns(["__index_level_0__"])
test_dataset = test_dataset.remove_columns(["__index_level_0__"])
# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'

train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataset.save_to_disk(training_input_path,fs=s3)

# save test_dataset to s3
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.save_to_disk(test_input_path,fs=s3)

# %%
from sagemaker.huggingface import HuggingFace

# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name': model_ckpt
                 }

huggingface_estimator = HuggingFace(entry_point='train.py',
                            source_dir='./scripts',
                            instance_type='ml.g4dn.xlarge',
                            instance_count=1,
                            role=role,
                            transformers_version='4.12',
                            pytorch_version='1.9',
                            py_version='py38',
                            hyperparameters = hyperparameters)

huggingface_estimator.fit({'train': training_input_path, 'test': test_input_path})
