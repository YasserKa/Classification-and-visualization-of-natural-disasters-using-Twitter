# %%

import ast
import itertools
import logging
from pprint import pprint

import nltk
import numpy as np
import pandas as pd
import visidata
from gensim import corpora
from gensim.models import LdaModel, Phrases
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path as abspath
from keybert import KeyBERT
from omegaconf import DictConfig
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flood_detection.data.preprocess import Language, Preprocess

LANGUAGE = Language.ENGLISH
preprocess = Preprocess(language=LANGUAGE)

# %%

with initialize_config_module(version_base=None, config_module="conf"):
    cfg: DictConfig = compose(config_name="config")

path_to_data = abspath(
    cfg.twitter_api.processed_geo + "_2021-08-17_to_2021-08-23__2022-11-25_15:26:36.csv"
)
# path_to_data = abspath(cfg.queensland.processed)

df_api = pd.read_csv(path_to_data)

path_to_data = abspath(cfg.supervisor.processed_flood)
df_sup = pd.read_csv(path_to_data)

df = df_api
df = df[df["predicted_label"] == 1].reset_index(drop=True).astype({"id": "str"})
# df = df[df["relevant"] == 1].reset_index(drop=True).astype({"id": "str"})

# %%

df = preprocess.clean_dataframe(df, translate=False)
# %%
# Extract tweets that only mention these locations
terms_locations_needed = ["Gävle", "Gävleborgs"]
df["swedish_locations_keys"] = df["swedish_locations"].apply(
    lambda x: list(ast.literal_eval(x).keys())
)
mask = df["swedish_locations_keys"].apply(
    lambda x: len(set(x).intersection(set(terms_locations_needed)))
)

df = df[mask > 0]

# %%
# Remove potential spam
# df = preprocess.get_one_tweet_for_each_user_per_week(df)


# Using cosine similarity
model = SentenceTransformer("all-MiniLM-L6-v2")


preprocess = Preprocess(language=Language.ENGLISH)
# df = df_[:30]
df["text_processed"] = preprocess.clean_text(df["text"].tolist())
df["embeddings"] = df["text_processed"].apply(
    lambda x: model.encode(x, convert_to_tensor=True)
)

df["similar"] = 0
for index, row in df.iterrows():
    for index1, row1 in df.iloc[index + 1 :].iterrows():
        cosine_scores = util.cos_sim(row["embeddings"], row1["embeddings"])
        if cosine_scores > 0.8:
            df.loc[index1, "similar"] = 1
df = df[df["similar"] != 1]

# %%

# NER

from flood_detection.predict.extract_location import Transform

swedish_ner_model = "KBLab/bert-base-swedish-cased-ner"
english_ner_model = "dslim/bert-base-NER"

model = Transform(english_ner_model)
from tqdm import tqdm

tqdm.pandas(desc="Extracting NER")
s = df["docs"].progress_apply(model.get_tokens)


# NOTE: can filter using score of entities
x = s.apply(
    lambda entities_row: [(entity["entity"], entity["word"]) for entity in entities_row]
)

b = x.explode().value_counts()

z = pd.DataFrame(data={"count": b, "b": b.index})
visidata.vd.view_pandas(z)

# %%

# Clean the text
if LANGUAGE == Language.SWEDISH:
    docs = preprocess.clean_text(df["text_raw"].tolist())
elif LANGUAGE == Language.ENGLISH:
    docs = preprocess.clean_text(df["text"].tolist())
else:
    raise Exception(f"{LANGUAGE.value} is not supported")


# docs = list(map(lambda x: preprocess.stem(x), docs))

# Apply lemmatization
docs = list(map(lambda x: preprocess.lemmatize(x), docs))

# Remove sentences with one term or less
docs = [[token for token in doc.split(" ") if len(token) > 1] for doc in docs]


# %%
docs = preprocess.clean_text(
    list(map(lambda x: " ".join(x), docs)),
    not_needed_words=["flood", "floods", "gävle"],
)
# docs = list(map(lambda x: x.split(" "), docs))

# %%
docs = list(filter(lambda x: len(x.split(" ")) > 4, docs))

# %%

import matplotlib
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Load the data and create document vectors
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(docs)

X_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=30
).fit_transform(X)


# %%

clustering = DBSCAN(eps=0.7, min_samples=3, metric="cosine").fit(X)
from collections import Counter

# print(X)
print(Counter(clustering.labels_))
# clustering = DBSCAN(eps=1.1, min_samples=2).fit(X)

matplotlib.pyplot.scatter(
    X_embedded[:, 0], X_embedded[:, 1], c=clustering.labels_, cmap="Set1", alpha=0.7
)

# %%


from sklearn.cluster import DBSCAN, KMeans

clusters = KMeans(n_clusters=5)
clusters.fit(X)


matplotlib.pyplot.scatter(
    X_embedded[:, 0], X_embedded[:, 1], c=clusters.labels_, cmap="Set1", alpha=0.7
)

# %%

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if "_" in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# %%
# TF-IDF scores
# from collections import defaultdict

from gensim import corpora

dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(text) for text in docs]

from gensim import models

tfidf_model = models.TfidfModel(corpus)  # step 1 -- initialize a model

# %%
corpus_tfidf = tfidf_model[corpus]

# for doc in corpus_tfidf:
#     print(doc)

import itertools

tfidf_list = list(itertools.chain.from_iterable(list(corpus_tfidf)))
# %%

df_tfidf = (
    pd.DataFrame(tfidf_list, columns=["term_id", "weight"])
    .groupby("term_id", as_index=False)
    .agg(
        weight_mean=pd.NamedAgg(column="weight", aggfunc="mean"),
        weight_count=pd.NamedAgg(column="term_id", aggfunc="count"),
    )
    .sort_values([("weight_count"), ("weight_mean")], ascending=False)
)
df_tfidf["term"] = df_tfidf["term_id"].apply(lambda x: dictionary[x])
print(df_tfidf)
# df_tfidf = (
#     pd.DataFrame(
#         [
#             sentence_keywords[0]
#             for sentence_keywords in extracted_keywords
#             if len(sentence_keywords) > 0
#         ],
#         columns=["keyword", "weight"],
#     )
#     .groupby("keyword", as_index=False)
#     .agg(
#         weight_mean=pd.NamedAgg(column="weight", aggfunc="mean"),
#         weight_count=pd.NamedAgg(column="keyword", aggfunc="count"),
#     )
#     .sort_values([("weight_count"), ("weight_mean")], ascending=False)
# )


# %%
lsi_model = models.LsiModel(
    corpus_tfidf, id2word=dictionary, num_topics=2
)  # initialize an LSI transformation
corpus_lsi = lsi_model[
    corpus_tfidf
]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

# %%

lsi_model.print_topics(1)
# %%

kw_model = KeyBERT()


docs_ = list(map(lambda x: " ".join(x), docs))
extracted_keywords = kw_model.extract_keywords(
    docs_, keyphrase_ngram_range=(1, 1), stop_words=None
)

df_bert = (
    pd.DataFrame(
        [
            sentence_keywords[0]
            for sentence_keywords in extracted_keywords
            if len(sentence_keywords) > 0
        ],
        columns=["keyword", "weight"],
    )
    .groupby("keyword", as_index=False)
    .agg(
        weight_mean=pd.NamedAgg(column="weight", aggfunc="mean"),
        weight_count=pd.NamedAgg(column="keyword", aggfunc="count"),
    )
    .sort_values([("weight_count"), ("weight_mean")], ascending=False)
)

print(df_bert)
# visidata.vd.view_pandas(df_bert)
# %%
dictionary = corpora.Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
not_below_perc = 0.05
min_docs = 20

dictionary.filter_extremes(
    no_below=min(min_docs, not_below_perc * len(docs)), no_above=0.85
)


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
# Keyword extraction
# https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

# Uses stopwords for english from NLTK, and all puntuation characters by
# default

nltk.data.path = ["./nltk_data"]
r = Rake()
# Extraction given the list of strings where each string is a sentence.
# r.extract_keywords_from_sentences(list(map(lambda x: " ".join(x), docs)))
r.extract_keywords_from_sentences(docs)

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()[:10]

# To get keyword phrases ranked highest to lowest with scores.
# r.get_ranked_phrases_with_scores()[:10]

# %%

n_gram_range = (5, 5)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(docs)

candidates = count.get_feature_names()

# %%

model = SentenceTransformer("distilbert-base-nli-mean-tokens")
doc_embedding = model.encode(docs)
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
