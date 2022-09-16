# %%
import datetime
import string
import re
import nltk
import numpy as np
import pandas as pd
import json

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

def clean_text(text):
    # Remove stopwords
    text = [word for word in text if word not in stopword]
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    # Remove URLs/Mentions/Hashtags/new lines
    text = re.sub('((www.[^s]+)|(https?://[^s]+)|(@[a-zA-Z0-9])|(#[a-zA-Z0-9])|\n)', ' ', text)
    # Remove Emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) 
    # Lower case
    text = text.lower()

    return text

df_needed['text_en'] = df_needed['text_en'].apply(lambda x: clean_text(x))

df_needed.head()

