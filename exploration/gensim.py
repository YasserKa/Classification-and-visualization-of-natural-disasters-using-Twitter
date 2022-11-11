# %%

import logging
from collections import defaultdict
from pprint import pprint

import pandas as pd
import spacy
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases

# %%
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path as abspath
from nltk.stem.wordnet import WordNetLemmatizer
from omegaconf import DictConfig

from flood_detection.data.preprocess import Preprocess

with initialize_config_module(version_base=None, config_module="conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = abspath(
    cfg.twitter_api.processed_flood + "_[2021, 8, 17]_to_[2021, 8, 23].json"
)

df = pd.read_csv(path_to_data)
# Manually annotated
df = df[df["predicted_label"] == 1]

# %%

# Clean the text
preprocess = Preprocess(language="sv")
docs = preprocess.clean_text(df["text_raw"].tolist())


# %%
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


nlp = spacy.load("sv_core_news_sm")
docs = list(map(lambda x: lemmatize(x), docs))

docs = [[token for token in doc if len(token) > 1] for doc in docs]

# %%
dictionary = corpora.Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

# %%
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
# Let’s see how many tokens and documents we have to train on.

print("Number of unique tokens: %d" % len(dictionary))
print("Number of documents: %d" % len(corpus))

# %%

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# %%
# Train LDA model.

# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = 0  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha="auto",
    eta="auto",
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
)

# %%
top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print("Average topic coherence: %.4f." % avg_topic_coherence)


pprint(top_topics)
