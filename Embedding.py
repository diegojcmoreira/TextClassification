from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np

import nltk


class MyTokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.vectors[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)

        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.wv.vectors[0])

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = nltk.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)

        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


def bag_of_word_embedding(data_frame):
    count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None,
                                       max_features=None)
    embedded_data_frame = count_vectorizer.fit_transform(data_frame['Text'])

    return embedded_data_frame


def tfidf_embedding(data_frame):
    count_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None,
                                       max_features=None)
    embedded_data_frame = count_vectorizer.fit_transform(data_frame['Text'])

    return embedded_data_frame

def w2v_embedding_mean(w2v_model, data_frame):
    mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2v_model)
    embedded_data_frame = mean_embedding_vectorizer.fit_transform(data_frame['Text'])

    return embedded_data_frame


def w2v_embedding_tfidf(w2v_model, data_frame):
    tfidf_embedding_vectorizer = TfidfEmbeddingVectorizer(w2v_model)
    embedded_data_frame = tfidf_embedding_vectorizer.fit_transform(data_frame['Text'])

    return embedded_data_frame

