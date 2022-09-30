# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
##############################################################################
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset


def prepare_data_avitm_federated(corpus, max_df, min_df):

    # Object that converts a collection of text documents into a matrix of token counts
    cv = CountVectorizer(input='content', lowercase=True, stop_words='english',
                         max_df=max_df, min_df=min_df, binary=False)
    docs = [" ".join(corpus[i]) for i in np.arange(len(corpus))]

    # Learn the vocabulary dictionary, train_bow = document-term matrix.
    train_bow = cv.fit_transform(docs).toarray()
    # Array mapping from feature integer indices to feature name.
    idx2token = cv.get_feature_names()
    train_dataset = BOWDataset(train_bow, idx2token)

    input_size = len(idx2token)

    id2token = {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}

    return train_dataset, input_size, id2token
