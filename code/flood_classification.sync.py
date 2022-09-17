# %%
import string
import re
import nltk
import pandas as pd
import json
import utils

# %%
with open(f"output/annotated_tweets_extracted.json", "r") as file:
    tweets_json = json.load(file)
df = pd.json_normalize(list(tweets_json.values()))

# %%
# NOTE: Some tweets, don't have geo-tagging or attachements, hence the NaN values
print(df.shape)
df.head()

# %%
df_needed = df[['id', 'text_en', 'On Topic']]
df_needed = df_needed[df_needed['On Topic'] != '']
df_needed=  df_needed.astype({'On Topic': 'int'})
df_needed["On Topic"].value_counts()

# %%
# Remove retweets
df_needed = df_needed[df_needed['text_en'].str[:2] != "RT"]
# Remove duplicates
df_needed = df_needed.drop_duplicates()

# %%

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

df_needed['text_en'] = df_needed['text_en'].apply(lambda x: utils.clean_text(x))

df_needed.head()

# %%
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df_needed)

dataset = dataset.add_column('label', dataset['On Topic'])
dataset = dataset.add_column('text', dataset['text_en'])
# dataset = dataset.add_column('input_ids', dataset['id'])

dataset = dataset.remove_columns(['On Topic', 'text_en', 'id'])

train_testvalid = dataset.train_test_split(test_size=0.1)

test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# %%
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2
model_ckpt = "bert-base-uncased"
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels).to(device))

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

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



batch_size = 16
logging_steps = len(train_test_valid_dataset_tokenized["train"]) // batch_size
model_name = f"bert-base-uncased-finetuned-floods"
torch.cuda.empty_cache()
# !PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10
training_args = TrainingArguments(output_dir=model_name,
        num_train_epochs=2,
        learning_rate=1e-2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.1,
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

