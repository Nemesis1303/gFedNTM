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


def prepare_data_avitm_federated(corpus, max_df, min_df):
    """It prepares the training data for each of the federated nodes in the format that is asked as input in AVITM.

    Args:
        * corpus (List[str]): List of documents
        * max_df (float / int): When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
        * min_df (float / int): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold

    Returns:
        # TODO: Revise types 
        * [List[BowDataset]]: List containing the training corpus for each node.
        * [List[tuple]]: List of mappings with the content of each training dataset's document-term matrix.
        * [List[tuple]]: List of the sizes of the input dimensions on the AVTIM models that are going to be trained at each node.
    """
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


def convert_topic_word_to_init_size(vocab_size, model, model_type,
                                     ntopics, id2token, all_words):
    """It converts the topic-word distribution matrix obtained from the training of a model into a matrix with the dimensions of the original topic-word distribution, assigning zeros to those words that are not present in the corpus. 
    It is only of use in case we are training a model over a synthetic dataset, so as to later compare the performance of the attained model in what regards to the similarity between the original and the trained model.

    Args:
        * vocab_size (int): Size of the synethic'data vocabulary.
        * model (AVITM): Model whose topic-word matrix is being transformed.
        * model_type (str): Type of the trained model (e.g. AVITM)
        * ntopics (int): Number of topics of the trained model.
        * id2token (List[tuple]): Mappings with the content of the document-term matrix.
        * all_words (List[str]): List of all the words of the vocabulary of size vocab_size.

    Returns:
        * [ndarray]: Normalized transormed topic-word distribution.
    """    
    if model_type == "avitm":
        w_t_distrib = np.zeros((10, vocab_size), dtype=np.float64)
        wd = model.get_topic_word_distribution()
        for i in np.arange(ntopics):
            for idx, word in id2token.items():
                for j in np.arange(len(all_words)):
                    if all_words[j] == word:
                        w_t_distrib[i, j] = wd[i][idx]
                        break
        sum_of_rows = w_t_distrib.sum(axis=1)
        normalized_array = w_t_distrib / sum_of_rows[:, np.newaxis]
        return normalized_array
    else:
        print("Method not impleemnted for the selected model type")
        return None
