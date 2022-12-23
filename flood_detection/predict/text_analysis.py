#!/usr/bin/env python

import itertools
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


class LDA_model:
    def __init__(self, docs, num_topics=1):
        self.docs = docs
        self.model = self.__train_model(docs, num_topics)
        self.num_topics = num_topics

    def get_topics(self, selected_docs):
        if self.model is None:
            return "Selected tweets is not enough"

        corpus = [self.dictionary.doc2bow(doc) for doc in selected_docs]

        terms_weights = [
            x[2]
            for x in list(self.model.get_document_topics(corpus, per_word_topics=True))
        ]
        terms_weights = list(itertools.chain.from_iterable(terms_weights))

        # Fill the weights for topics with 0 for each term if they don't exist
        terms_weights = [
            [x[0]]
            + [[0] * y[0] + [y[1]] + [0] * (self.num_topics - y[0] - 1) for y in x[1]]
            for x in terms_weights
        ]
        terms_weights = [[x[0]] + x[1] for x in terms_weights]

        columns = ["term_id"] + [f"topic_{x}" for x in range(self.num_topics)]

        df_terms_weights = (
            pd.DataFrame(terms_weights, columns=columns)
            .groupby("term_id", as_index=False)
            .agg(
                count=pd.NamedAgg(column="term_id", aggfunc="count"),
                **{
                    f"topic_{x}_mean": pd.NamedAgg(column=f"topic_{x}", aggfunc="mean")
                    for x in range(self.num_topics)
                },
            )
            .sort_values([("count")], ascending=False)
        )
        df_terms_weights["term"] = df_terms_weights["term_id"].apply(
            lambda x: self.dictionary[x]
        )
        return df_terms_weights

    def __train_model(self, docs, num_topics=1):

        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(docs, min_count=20)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if "_" in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        self.dictionary = corpora.Dictionary(docs)

        # Filter out words that occur less than min(20 documents, 5% in documents), or more than 75% of the documents.
        not_below_perc = 0.05
        min_docs = 20

        self.dictionary.filter_extremes(
            no_below=min(min_docs, not_below_perc * len(docs)), no_above=0.75
        )

        # Bag-of-words representation of the documents.
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        # the corpus isn't enough to train LDA model
        if len(self.dictionary) == 0:
            return None

        # Set training parameters.
        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = 0  # Don't evaluate model perplexity, takes too much time.

        # Make an index to word dictionary.
        temp = self.dictionary[0]  # type: ignore # This is only to "load" the dictionary.
        id2word = self.dictionary.id2token

        model = LdaModel(
            corpus=self.corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha="auto",
            eta="auto",
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
        )

        return model


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
            lda_model = LDA_model(docs)
            output = lda_model.get_topics(docs)
        case _:
            raise Exception(f"{text_analysis_method} not available")

    pprint(output)


if __name__ == "__main__":
    main()
