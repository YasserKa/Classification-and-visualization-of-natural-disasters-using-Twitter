# %%

import itertools
import logging
import os
from pprint import pprint

import numpy as np
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path as abspath
from nltk.stem.snowball import SnowballStemmer
from omegaconf import DictConfig
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flood_detection.data.preprocess import Preprocess

# %%

with initialize_config_module(version_base=None, config_module="conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = abspath(
    cfg.twitter_api.processed_flood + "_2021-08-17_to_2021-08-23.csv"
)
df_api = pd.read_csv(path_to_data)

path_to_data = abspath(cfg.supervisor.processed_flood)
df_sup = pd.read_csv(path_to_data)

df = df_api
df = df[df["predicted_label"] == 1]

language = "en"

preprocess = Preprocess(language=language)

# Remove potential spam
df = preprocess.get_one_tweet_for_each_user_per_week(df)
# %%

# Clean the text
if language == "sv":
    docs = preprocess.clean_text(df["text_raw"].tolist())
    nlp = spacy.load("sv_core_news_sm")
    stemmer = SnowballStemmer(language="swedish")
elif language == "en":
    docs = preprocess.clean_text(df["text_translated"].tolist())
    nlp = spacy.load("en_core_web_sm")
    stemmer = SnowballStemmer(language="english")
else:
    raise Exception(f"{language} is not supported")


# %%
def stem(sentence):
    return " ".join([stemmer.stem(word) for word in sentence.split(" ")])


docs = list(map(lambda x: stem(x), docs))


# %%
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


docs = list(map(lambda x: lemmatize(x), docs))

docs = [[token for token in doc if len(token) > 1] for doc in docs]

# %%
docs = preprocess.clean_text(list(map(lambda x: " ".join(x), docs)))
docs = list(map(lambda x: x.split(" "), docs))

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
num_topics = 1
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

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print("Average topic coherence: %.4f." % avg_topic_coherence)


pprint(top_topics)

# %%
docs_without_flood = list(map(lambda x: list(filter(lambda y: y != "flood", x)), docs))

docs_without_flood_sentences = list(map(lambda x: " ".join(x), docs_without_flood))

# %%

# Keyword extraction
# https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

# Uses stopwords for english from NLTK, and all puntuation characters by
# default

os.environ["NLTK_DATA"] = "./nltk_data"
r = Rake()
# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences(list(map(lambda x: " ".join(x), docs)))
r.extract_keywords_from_sentences(docs_without_flood_sentences)

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()[:10]

# To get keyword phrases ranked highest to lowest with scores.
# r.get_ranked_phrases_with_scores()[:10]

# %%

n_gram_range = (5, 5)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(
    docs_without_flood_sentences
)

candidates = count.get_feature_names()

# %%

model = SentenceTransformer("distilbert-base-nli-mean-tokens")
doc_embedding = model.encode(docs_without_flood_sentences)
candidate_embeddings = model.encode(candidates)

# %%


top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

# %%


def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum(
            [
                distances_candidates[i][j]
                for i in combination
                for j in combination
                if i != j
            ]
        )
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=10, nr_candidates=20)
