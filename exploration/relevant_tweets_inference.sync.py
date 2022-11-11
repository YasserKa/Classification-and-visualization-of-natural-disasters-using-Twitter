# %%
from transformers import DistilBertTokenizer
from datasets.arrow_dataset import Dataset
from hydra import compose, initialize
from omegaconf import DictConfig
import pandas as pd
from transformers import DistilBertForSequenceClassification
import torch
from torch.nn.functional import cross_entropy

# %%
PATH_TO_MODEL = "../models"

with initialize(version_base=None, config_path="../conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed


df1 = pd.read_csv("../" + path_to_data)
dataset: Dataset = Dataset.from_pandas(df1)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(
    f"{PATH_TO_MODEL}/vocab.txt", local_files_only=True
)
model = DistilBertForSequenceClassification.from_pretrained(
    f"{PATH_TO_MODEL}/pytorch_model.bin",
    config=f"{PATH_TO_MODEL}/config.json",
    local_files_only=True,
).to(device)


# %%
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


tokenized = dataset.map(tokenize, batched=True, batch_size=None)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "relevant"])


# %%
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(
            output.logits, batch["relevant"].to(device), reduction="none"
        )
        # Place outputs on CPU for compatibility with other dataset columns
        return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}


tokenized = tokenized.map(forward_pass_with_label, batched=True, batch_size=16)
tokenized.set_format("pandas")
# %%
pd.options.display.max_colwidth = 100
cols = ["text", "relevant", "predicted_label", "loss"]
test = tokenized[:][cols]
test[test["relevant"] == 1].sort_values("loss", ascending=False).head(100)
