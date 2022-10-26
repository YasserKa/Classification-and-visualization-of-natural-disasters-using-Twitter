# %%

import logging
from collections import defaultdict
from pprint import pprint

import nltk
import pandas as pd
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases
from hydra import compose, initialize
from nltk.stem.wordnet import WordNetLemmatizer
from omegaconf import DictConfig

# %%

with initialize(version_base=None, config_path="conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = cfg.supervisor.processed

df = pd.read_csv(path_to_data)
# Use NER on only relevant tweets
df = df[df["relevant"] == 1]

# %%
# Count word frequencies

frequency = defaultdict(int)
for text in df["raw_text"].values:
    for token in text.split():
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [
    [token for token in text.split() if frequency[token] > 1]
    for text in df["raw_text"].values
]


# %%

dictionary = corpora.Dictionary(processed_corpus)

# %%
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# train the model
tfidf = models.TfidfModel(bow_corpus)

# %%

# index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)


# query_document = "flooding".split()
# query_bow = dictionary.doc2bow(query_document)
# sims = index[tfidf[query_bow]]
# print(list(enumerate(sims)))

# %%
# Lemmatize the documents.

nltk.download("wordnet", download_dir="./nltk_data")
nltk.download("omw-1.4", download_dir="./nltk_data")
lemmatizer = WordNetLemmatizer()
docs = [
    [lemmatizer.lemmatize(token) for token in doc]
    for doc in df["raw_text"].apply(lambda x: x.split()).values
]

# %%
# Compute bigrams.


# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if "_" in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
# %%

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

# %%
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
# Let’s see how many tokens and documents we have to train on.

print("Number of unique tokens: %d" % len(dictionary))
print("Number of documents: %d" % len(corpus))

# %%
# Train LDA model.

# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

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

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# %%
top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print("Average topic coherence: %.4f." % avg_topic_coherence)


pprint(top_topics)

# %%
print(df)
