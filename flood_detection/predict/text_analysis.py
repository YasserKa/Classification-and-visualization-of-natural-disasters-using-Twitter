#!/usr/bin/env python

from enum import Enum
from pprint import pprint

import click
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, Phrases
from hydra.utils import to_absolute_path as abspath

from flood_detection.data.preprocess import Language, Preprocess


class TextAnalysisMethod(Enum):
    LDA = "lda"


def get_preprocessed_docs(df, language):
    preprocess = Preprocess(language)

    # Clean the text
    if language == Language.SWEDISH:
        docs = preprocess.clean_text(df["text_raw"].tolist())
    elif language == Language.ENGLISH:
        docs = preprocess.clean_text(df["text"].tolist())
    else:
        raise Exception(f"{language.value} is not supported")

    docs = list(map(lambda x: preprocess.lemmatize(x), docs))

    docs = [[token for token in doc.split(" ") if len(token) > 1] for doc in docs]

    docs = preprocess.clean_text(
        list(map(lambda x: " ".join(x), docs)),
        ["flood", "gävle", "gävleborgs", "alberta"],
    )
    docs = list(map(lambda x: x.split(" "), docs))

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if "_" in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


def perform_LDA(docs):
    dictionary = corpora.Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    if len(dictionary) == 0:
        return "The corpus isn't enough"

    # Set training parameters.
    num_topics = 1
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = 0  # Don't evaluate model perplexity, takes too much time.

    # Make an index to word dictionary.
    temp = dictionary[0]  # type: ignore # This is only to "load" the dictionary.
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

    return top_topics


@click.command()
@click.option(
    "--text_analysis_method",
    default=TextAnalysisMethod.LDA.value,
    help="Text analysis method used (lda)",
)
@click.option(
    "--language",
    default=Language.ENGLISH.value,
    help="Language used in dataset (english, swedish)",
)
@click.argument("path_to_data", nargs=-1)
def main(text_analysis_method, language, path_to_data):
    input_path: str = abspath(path_to_data[0])
    df = pd.read_csv(input_path)

    df = df[df["predicted_label"] == 1].reset_index(drop=True).astype({"id": "str"})

    language = Language(language)
    text_analysis_method = TextAnalysisMethod(text_analysis_method)
    docs = get_preprocessed_docs(df, language)

    match text_analysis_method:
        case TextAnalysisMethod.LDA:
            output = perform_LDA(docs)
        case _:
            raise Exception(f"{text_analysis_method} not available")

    pprint(output)


if __name__ == "__main__":
    main()
