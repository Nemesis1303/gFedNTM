# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                                UTILS                                   ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils.bow_dataset import BOWDataset


def prepare_data_avitm_federated(n_nodes, corpus_nodes, max_df, min_df):
    """It prepares the training data for each of the federated nodes in the format that is asked as input in AVITM.

    Args:
        * n_nodes (int): Number of nodes (i.e. clients) in the federated scenario
        * corpus_nodes (List[str]): List of corpora, each of them corresponding to the training corpus of each of the nodes in the federated scenario.
        * max_df (float / int): When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
        * min_df (float / int): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold

    Returns:
        [List[BowDataset]]: List containing the training corpus for each node.
        [List[ndarray]]: List of mappings with the content of each training dataset's document-term matrix.
        [List[tuple]]: List of the sizes of the input dimensions on the AVTIM models that are going to be trained at each node.
    """
    train_datasets = []
    id2tokens = []
    input_sizes = []
    for node in n_nodes:
        # Object that converts a collection of text documents into a matrix of token counts
        cv = CountVectorizer(input='content', lowercase=True, stop_words='english',
                             max_df=max_df, min_df=min_df, binary=False)
        corpus_node = corpus_nodes[node]
        docs = [" ".join(corpus_node[i]) for i in np.arange(len(corpus_node))]

        # Learn the vocabulary dictionary, train_bow = document-term matrix.
        train_bow = cv.fit_transform(docs).toarray()
        # Array mapping from feature integer indices to feature name.
        idx2token = cv.get_feature_names()
        train_dataset = BOWDataset(train_bow, idx2token)
        train_datasets.append(train_dataset)

        input_size = len(idx2token)
        input_sizes.append(input_size)

        id2token = {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}
        id2tokens.append(id2token)

    return train_datasets, id2tokens, input_sizes
